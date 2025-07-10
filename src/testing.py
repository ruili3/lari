import argparse
import datetime
import json
import math
import os
import sys
import time
import random
from collections import defaultdict
from pathlib import Path
import torchvision.transforms as transforms

import numpy as np
import torch
import torch.backends.cudnn as cudnn
torch.backends.cuda.matmul.allow_tf32 = True  # for gpu >= Ampere and pytorch >= 1.12

import wandb

# Import modules from your project
from src.lari.model import LaRIModel  # noqa: F401, needed when loading the model
from src.datasets import get_data_loader  # noqa
from src.metrics import *  # noqa: F401, needed when loading the model
from src.inference import loss_of_one_batch_eval  # noqa
import src.utils.misc as misc  # noqa
from src.utils.vis import denormalize, save_point_cloud

def get_args_parser():
    parser = argparse.ArgumentParser('LaRI Testing', add_help=False)
    # Experiment / logging info
    parser.add_argument('--proj_name', default="lapt", type=str,
                        help="experiment name for wandb logging")
    parser.add_argument('--exp_name', default=None, type=str,
                        help="experiment name for wandb logging")
    # Model and criterion
    parser.add_argument('--model', default=None,
                        type=str, help="string containing the model to build")
    parser.add_argument('--pretrained',
                        help='Path of a starting checkpoint')
    parser.add_argument('--test_criterion', default=None, type=str,
                        help="Test criterion")
    # Dataset
    parser.add_argument('--test_dataset', default='[None]', type=str,
                        help="Testing set. For multiple datasets, separate names with a plus sign (e.g., dataset1+dataset2)")
    # Misc. settings
    parser.add_argument('--seed', default=0, type=int, help="Random seed")
    parser.add_argument('--batch_size', default=64, type=int,
                        help="Batch size per GPU")
    parser.add_argument('--amp', type=int, default=0, choices=[0, 1],
                        help="Use Automatic Mixed Precision for testing")
    parser.add_argument("--disable_cudnn_benchmark", action='store_true', default=False,
                        help="set cudnn.benchmark = False")
    # Distributed / parallel settings
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--world_size', default=1, type=int,
                        help='Number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_url', default='env://',
                        help='URL used to set up distributed testing')
    # Evaluation / logging frequency
    parser.add_argument('--print_freq', default=20, type=int,
                        help='Frequency (in iterations) to print testing info')
    # Visualization settings
    parser.add_argument('--save_3dpts_per_n_batch', default=10, type=int,
                        help='Number of saved samples for visualization')
    # Output directories
    parser.add_argument('--output_dir', default=None, type=str,
                        help="Path where to save the output")
    parser.add_argument('--wandb_dir', default=None, type=str,
                        help="Path where to save the wandb results")

    return parser


def build_dataset(dataset, batch_size, num_workers, test=True):
    split = ['Train', 'Test'][test]
    print(f'Building {split} Data loader for dataset: {dataset}')
    loader = get_data_loader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_mem=True,
        shuffle=not test,
        drop_last=not test
    )
    print(f"{split} dataset length: {len(loader)}")
    return loader


@torch.no_grad()
def test_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                   data_loader, device: torch.device, epoch: int,
                   args, write_log=False, prefix='test', is_main_proc=False):
    """
    Run a single testing epoch.
    """
    model.eval()
    metric_logger = misc.BSAgonisticMetricLogger(delimiter="  ")
    metric_logger.meters = defaultdict(lambda: misc.SmoothedValue(window_size=9**9))
    header = f'Test Epoch: [{epoch}]'
    
    # Set epoch for distributed sampling (if applicable)
    if hasattr(data_loader, 'dataset') and hasattr(data_loader.dataset, 'set_epoch'):
        data_loader.dataset.set_epoch(epoch)
    if hasattr(data_loader, 'sampler') and hasattr(data_loader.sampler, 'set_epoch'):
        data_loader.sampler.set_epoch(epoch)
    

    for i, batch in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
        
        loss_tuple, pred_dict = loss_of_one_batch_eval(batch, model, criterion, device,
                                                use_amp=bool(args.amp))
        sampled_pts3d_pred_gt, loss_details = loss_tuple
        
        # log metrics
        metric_logger.update(**loss_details)

        # save visualizations
        if i % args.save_3dpts_per_n_batch == 0:

            if args.output_dir and is_main_proc:
                name = batch["name"][0].replace("/","_")
                # both in B N 3
                pts3d_pred, pts3d_gt, _ = sampled_pts3d_pred_gt

                os.makedirs(os.path.join(args.output_dir, "plys"), exist_ok=True)
                pred_filename = os.path.join(args.output_dir, "plys", f"{name}_pred.ply")
                gt_filename = os.path.join(args.output_dir, "plys", f"{name}_gt.ply")
                img_filename = os.path.join(args.output_dir, "plys", f"{name}_rgb.jpg")
                # save plys in different color
                save_pred_gt_point_clouds(pts3d_pred[0], pts3d_gt[0], batch["img"][0].unsqueeze(0), pred_filename, gt_filename, img_filename)


    # Gather and print stats across processes (if using distributed evaluation)
    metric_logger.synchronize_between_processes()
    print("Averaged testing stats:", metric_logger)
    aggs = [('avg', 'global_avg'), ('med', 'median')]
    results = {f'{k}_{tag}': getattr(meter, attr)
               for k, meter in metric_logger.meters.items()
               for tag, attr in aggs}
    
    if write_log:
        for name, val in results.items():
            wandb.log({f"{prefix}_{name}": val}, step=0)

    return results





def save_pred_gt_point_clouds(pred, gt, img, pred_filename, gt_filename, rgb_filename):
    """
    Save predicted and ground truth point clouds with different colors.
    """
    # Convert to numpy
    pred_np = pred.cpu().numpy() if isinstance(pred, torch.Tensor) else pred
    gt_np = gt.cpu().numpy() if isinstance(gt, torch.Tensor) else gt

    # Assign colors (pred: blue, gt: red)
    pred_rgb = np.tile(np.array([[0, 0, 255]], dtype=np.uint8), (pred_np.shape[0], 1))  # Blue
    gt_rgb = np.tile(np.array([[255, 0, 0]], dtype=np.uint8), (gt_np.shape[0], 1))  # Red

    # Save point clouds
    save_point_cloud(pred_np, pred_rgb, pred_filename)
    save_point_cloud(gt_np, gt_rgb, gt_filename)

    if img is not None:
        # image
        img = denormalize(img).squeeze()
        img = torch.clip(img, min=0, max=1.0)
        img = transforms.ToPILImage()(img)
        img.save(rgb_filename)



def test(args):
    random.seed(777)
    # Set the random seed for reproducibility.
    torch.manual_seed(777)
    torch.cuda.manual_seed_all(777)
    # Set NumPy seed
    np.random.seed(777)

    # Initialize distributed mode and device
    misc.init_distributed_mode(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cudnn.benchmark = not args.disable_cudnn_benchmark

    # Create output directory if needed
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Build test dataset(s)
    print('Building test dataset(s):', args.test_dataset)
    data_loader_test = {
        dataset.split('(')[0]: build_dataset(dataset, args.batch_size, args.num_workers, test=True)
        for dataset in args.test_dataset.split('+')
    }
    
    # Load the model
    print('Loading model:', args.model)
    model = eval(args.model)
    model.to(device)
    print("Model architecture:\n", model)
    
    # Create test criterion
    criterion_str = args.test_criterion
    print(f'Using test criterion: {criterion_str}')
    test_criterion = eval(criterion_str).to(device)
    
    # Load pretrained weights if provided
    if args.pretrained is not None:
        print('Loading pretrained model from:', args.pretrained)
        ckpt = torch.load(args.pretrained, map_location=device)
        if 'model' in ckpt:
            model.load_state_dict(ckpt['model'], strict=False)
        else:
            model.load_state_dict(ckpt, strict=False)

    # Optionally initialize wandb for logging
    if misc.is_main_process() and args.wandb_dir is not None:
        wandb.init(
            project=args.proj_name,
            name=args.exp_name if args.exp_name else None,
            config=vars(args),
            dir=args.wandb_dir,
        )
    
    # Run testing on each dataset
    all_results = {}  # Dictionary to store metrics for all datasets
    for test_name, test_loader in data_loader_test.items():
        print(f"\nTesting on dataset: {test_name}")
        stats = test_one_epoch(model, test_criterion, test_loader,
                            device, epoch=0, args=args, write_log=(args.wandb_dir is not None) and misc.is_main_process(), prefix=test_name,
                            is_main_proc=misc.is_main_process())
        print(f"Results for {test_name} dataset: {stats}")
        all_results[test_name] = stats  # Save the results for this dataset

    # After testing all datasets, save the aggregated metrics to a JSON file.
    results_path = os.path.join(args.output_dir, "test_metrics.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=4)
    print(f"Saved test metrics to {results_path}")


def main():
    parser = get_args_parser()
    args = parser.parse_args()
    test(args)


if __name__ == '__main__':
    main()
