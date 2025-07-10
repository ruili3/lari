import argparse
import datetime
import json
import numpy as np
import os
import sys
import time
import math
from collections import defaultdict
from pathlib import Path
from typing import Sized

import torch
import torch.backends.cudnn as cudnn
# from torch.utils.tensorboard import SummaryWriter
torch.backends.cuda.matmul.allow_tf32 = True  # for gpu >= Ampere and pytorch >= 1.12

from src.lari.model import LaRIModel  # noqa: F401, needed when loading the model
from src.datasets import get_data_loader  # noqa
from src.losses import *  # noqa: F401, needed when loading the model
from src.metrics import SSI3DScore_Object, SSI3DScore_Scene
from src.inference import loss_of_one_batch  # noqa

import src.utils.misc as misc  # noqa
from src.utils.misc import NativeScalerWithGradNormCount as NativeScaler  # noqa
import wandb
from src.utils.vis import make_wandb_vis, prob_to_mask

def get_args_parser():
    parser = argparse.ArgumentParser('LaRI training', add_help=False)
    parser.add_argument('--proj_name', default="debug",
                        type=str, help="experiment name for wandb logging")
    parser.add_argument('--exp_name', default="debug",
                        type=str, help="experiment name for wandb logging")
    # model and criterion
    parser.add_argument('--model', default=None,
                        type=str, help="string containing the model to build")
    parser.add_argument('--pretrained', default=None, help='path of a starting checkpoint')
    parser.add_argument('--train_criterion', default=None,
                        type=str, help="train criterion")
    parser.add_argument('--test_criterion', default=None, type=str, help="test criterion")

    parser.add_argument('--model_type', type=str, default="unet", help="network architecture type, used to identify inference patterns")


    # dataset
    parser.add_argument('--train_dataset', required=True, type=str, help="training set")
    parser.add_argument('--test_dataset', default='[None]', type=str, help="testing set")

    # training
    parser.add_argument('--seed', default=0, type=int, help="Random seed")
    parser.add_argument('--batch_size', default=24, type=int,
                        help="Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus")
    parser.add_argument('--accum_iter', default=1, type=int,
                        help="Accumulate gradient iterations (for increasing the effective batch size under memory constraints)")
    parser.add_argument('--epochs', default=800, type=int, help="Maximum number of epochs for the scheduler")

    parser.add_argument('--weight_decay', type=float, default=0.05, help="weight decay (default: 0.05)")
    parser.add_argument('--lr', type=float, default=None, metavar='LR', help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1.5e-4, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N', help='epochs to warmup LR')

    parser.add_argument('--amp', type=int, default=0,
                        choices=[0, 1], help="Use Automatic Mixed Precision for pretraining")
    parser.add_argument("--disable_cudnn_benchmark", action='store_true', default=False,
                        help="set cudnn.benchmark = False")
    # others
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    parser.add_argument('--eval_freq', type=int, default=1, help='Test loss evaluation frequency')
    parser.add_argument('--save_freq', default=1, type=int,
                        help='frequence (number of epochs) to save checkpoint in checkpoint-last.pth')
    parser.add_argument('--keep_freq', default=20, type=int,
                        help='frequence (number of epochs) to save checkpoint in checkpoint-%d.pth')
    parser.add_argument('--print_freq', default=20, type=int,
                        help='frequence (number of iterations) to print infos while training')

    # visualization
    parser.add_argument('--n_save_intermediate', default=2, type=int,
                    help='number of saved samples for visualization')
    parser.add_argument('--n_vis_layers', default=5, type=int,
                        help='number of LaRI layers stored for visualization')
    parser.add_argument('--n_vis_pts3d', default=10000, type=int,
                        help='number of points saved for each sample')

    # output dir
    parser.add_argument('--output_dir', default='./output/', type=str, help="path where to save the output")
    parser.add_argument('--wandb_dir', default='./wandb/', type=str, help="path where to save the wandb results")


    parser.add_argument('--clip_grad_10', action="store_true", default=False)

    return parser


def train(args):
    misc.init_distributed_mode(args)
    global_rank = misc.get_rank()
    world_size = misc.get_world_size()

    print("output_dir: " + args.output_dir)
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # auto resume
    last_ckpt_fname = os.path.join(args.output_dir, f'checkpoint-last.pth')
    args.resume = last_ckpt_fname if os.path.isfile(last_ckpt_fname) else None

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    # fix the seed
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = not args.disable_cudnn_benchmark

    # training dataset and loader
    print('Building train dataset {:s}'.format(args.train_dataset))
    #  dataset and loader
    data_loader_train = build_dataset(args.train_dataset, args.batch_size, args.num_workers, test=False)
    print('Building test dataset {:s}'.format(args.train_dataset))
    data_loader_test = {dataset.split('(')[0]: build_dataset(dataset, args.batch_size, args.num_workers, test=True)
                        for dataset in args.test_dataset.split('+')}

    # model
    print('Loading model: {:s}'.format(args.model))
    model = eval(args.model)
    print(f'>> Creating train criterion = {args.train_criterion}')
    train_criterion = eval(args.train_criterion).to(device)
    print(f'>> Creating test criterion = {args.test_criterion or args.train_criterion}')
    test_criterion = eval(args.test_criterion or args.criterion).to(device)

    model.to(device)
    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

    # Count total parameters: 314.232M
    total_params = sum(p.numel() for p in model.parameters())
    print("MODEL parameters: {:.3f}".format(total_params / 1e6))


    if args.pretrained and not args.resume:
        print('Loading pretrained: ', args.pretrained)
        ckpt = torch.load(args.pretrained, map_location=device)
        print(model.load_state_dict(ckpt['model'], strict=False))
        del ckpt  # in case it occupies memory

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256
    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)
    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True, static_graph=True)
        model_without_ddp = model.module

    # following timm: set wd as 0 for bias and norm layers
    param_groups = misc.get_parameter_groups(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()

    def write_log_stats(epoch, train_stats, test_stats):
        if misc.is_main_process():

            log_stats = dict(epoch=epoch, **{f'train_{k}': v for k, v in train_stats.items()})
            for test_name in data_loader_test:
                if test_name not in test_stats:
                    continue
                log_stats.update({test_name + '_' + k: v for k, v in test_stats[test_name].items()})

            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    def save_model(epoch, fname, best_so_far):
        misc.save_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer,
                        loss_scaler=loss_scaler, epoch=epoch, fname=fname, best_so_far=best_so_far)
    
    # load model parameters & loss_scalers & optimizers for resuming the training process
    best_so_far = misc.load_model(args=args, model_without_ddp=model_without_ddp,
                                  optimizer=optimizer, loss_scaler=loss_scaler)
    if best_so_far is None:
        best_so_far = float('inf')
        
    if global_rank == 0 and args.output_dir is not None:
        write_log = True
        wandb.init(
            project=args.proj_name,
            name=args.exp_name,
            config=vars(args),
            dir = args.wandb_dir,
        )
    else:
        write_log = False

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    train_stats = test_stats = {}
    for epoch in range(args.start_epoch, args.epochs + 1):

        # Save immediately the last checkpoint
        if epoch > args.start_epoch:
            if args.save_freq and epoch % args.save_freq == 0 or epoch == args.epochs:
                save_model(epoch - 1, 'last', best_so_far)

        # Test on multiple datasets
        new_best = False
        if (epoch > 0 and args.eval_freq > 0 and epoch % args.eval_freq == 0):
            test_stats = {}
            for test_name, testset in data_loader_test.items():
                stats = test_one_epoch(model, test_criterion, testset,
                                       device, epoch, write_log=write_log, args=args, prefix=test_name)
                test_stats[test_name] = stats

                # Save best of all
                if stats['loss_med'] < best_so_far:
                    best_so_far = stats['loss_med']
                    new_best = True

        # Save more stuff
        write_log_stats(epoch, train_stats, test_stats)

        if epoch > args.start_epoch:
            if args.keep_freq and epoch % args.keep_freq == 0:
                save_model(epoch - 1, str(epoch), best_so_far)
            if new_best:
                save_model(epoch - 1, 'best', best_so_far)
        if epoch >= args.epochs:
            break  # exit after writing last test to disk

        # Train
        train_stats = train_one_epoch(
            model, train_criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            write_log=write_log,
            args=args)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    save_final_model(args, args.epochs, model_without_ddp, best_so_far=best_so_far)


def save_final_model(args, epoch, model_without_ddp, best_so_far=None):
    output_dir = Path(args.output_dir)
    checkpoint_path = output_dir / 'checkpoint-final.pth'
    to_save = {
        'args': args,
        'model': model_without_ddp if isinstance(model_without_ddp, dict) else model_without_ddp.cpu().state_dict(),
        'epoch': epoch
    }
    if best_so_far is not None:
        to_save['best_so_far'] = best_so_far
    print(f'>> Saving model to {checkpoint_path} ...')
    misc.save_on_master(to_save, checkpoint_path)


def build_dataset(dataset, batch_size, num_workers, test=False):
    split = ['Train', 'Test'][test]
    print(f'Building {split} Data loader for dataset: ', dataset)
    loader = get_data_loader(dataset,
                             batch_size=batch_size,
                             num_workers=num_workers,
                             pin_mem=True,
                             shuffle=not (test),
                             drop_last=not (test))

    print(f"{split} dataset length: ", len(loader))
    return loader


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Sized, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    args,
                    write_log=False):
    assert torch.backends.cuda.matmul.allow_tf32 == True

    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    accum_iter = args.accum_iter


    if hasattr(data_loader, 'dataset') and hasattr(data_loader.dataset, 'set_epoch'):
        data_loader.dataset.set_epoch(epoch)
    if hasattr(data_loader, 'sampler') and hasattr(data_loader.sampler, 'set_epoch'):
        data_loader.sampler.set_epoch(epoch)

    optimizer.zero_grad()

    for data_iter_step, batch in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
        epoch_f = epoch + data_iter_step / len(data_loader)

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            misc.adjust_learning_rate(optimizer, epoch_f, args)

        loss_tuple, pred_dict = loss_of_one_batch(batch, model, criterion, device,
                                       use_amp=bool(args.amp), model_type=args.model_type, is_eval=False)
        loss, loss_details = loss_tuple  # criterion returns two values
        loss_value = float(loss)

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value), force=True)
            sys.exit(1)

        loss /= accum_iter

        loss_scaler(loss, optimizer, clip_grad=None if not args.clip_grad_10 else 10.0,
                    parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)


        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        del loss
        

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(epoch=epoch_f)
        metric_logger.update(lr=lr)
        metric_logger.update(loss=loss_value, **loss_details)

        if (data_iter_step + 1) % accum_iter == 0 and ((data_iter_step + 1) % (accum_iter * args.print_freq)) == 0:
            loss_value_reduce = misc.all_reduce_mean(loss_value)  # MUST BE EXECUTED BY ALL NODES
            if not write_log:
                continue
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int(epoch_f * 1000)
            wandb.log({'train_loss': loss_value_reduce}, step=epoch_1000x)
            wandb.log({'train_lr': lr}, step=epoch_1000x)
            wandb.log({'train_iter': epoch_1000x}, step=epoch_1000x)

            for name, val in loss_details.items():
                wandb.log({f'train_{name}': val}, step=epoch_1000x)

            # save visualizations
            save_vis(batch, pred_dict, epoch_1000x, args.n_save_intermediate, args.n_vis_layers, args.n_vis_pts3d, mode="tr")

        del batch
        del pred_dict

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def save_vis(batch, pred_dict, epoch, n_save_intermediate=1, n_vis_layer=4, n_vis_pts3d=5000, mode="tr"):
    '''
    save intermediate results of specified samples from the batch and prediction results.
    '''

    n_save_intermediate = min(n_save_intermediate, batch["img"].shape[0])


    n_pred_layer = pred_dict["pts3d"].shape[-2]
    n_vis_layer = min(n_vis_layer, n_pred_layer)

    for n in range(n_save_intermediate):
        image = batch["img"][n,...]
        pts3d_gt = batch["pts3d"][n,...]
        mask_gt = batch["mask"][n,...]

        pts3d_pred = pred_dict["pts3d"][n,...]

        if 'mask' in pred_dict or 'seg_prob' in pred_dict:
            mask_pred = (pred_dict["mask"][n,...] > 0.5).float() if "mask" in pred_dict else prob_to_mask(pred_dict["seg_prob"][n,...])
        else:
            mask_pred = torch.zeros(mask_gt.shape)


        res_image_np, pred_pts3d_sampled, gt_pts3d_sampled, pred_pts3d_unmasked = make_wandb_vis(image, 
                                                                                                    pts3d_gt, 
                                                                                                    pts3d_pred, 
                                                                                                    valid_mask=mask_gt,
                                                                                                    pred_mask=mask_pred,
                                                                                                    n_vis_layer=n_vis_layer,
                                                                                                    n_3dpts=n_vis_pts3d
                                                                                                    )

        wandb.log({f"{mode}_res_{n}": wandb.Image(res_image_np)}, step=epoch) # image saving
        wandb.log({f"{mode}_3d_pred_{n}": wandb.Object3D(pred_pts3d_sampled)}, step=epoch) # point cloud saving
        wandb.log({f"{mode}_3d_gt_{n}": wandb.Object3D(gt_pts3d_sampled)}, step=epoch)
        wandb.log({f"{mode}_3d_pred_unmasked_{n}": wandb.Object3D(pred_pts3d_unmasked)}, step=epoch)

    return


@torch.no_grad()
def test_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                   data_loader: Sized, device: torch.device, epoch: int,
                   args, write_log=False, prefix='test'):

    model.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.meters = defaultdict(lambda: misc.SmoothedValue(window_size=9**9))
    header = 'Test Epoch: [{}]'.format(epoch)


    if hasattr(data_loader, 'dataset') and hasattr(data_loader.dataset, 'set_epoch'):
        data_loader.dataset.set_epoch(epoch)
    if hasattr(data_loader, 'sampler') and hasattr(data_loader.sampler, 'set_epoch'):
        data_loader.sampler.set_epoch(epoch)

    for i, batch in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
        loss_tuple, pred_dict = loss_of_one_batch(batch, model, criterion, device,
                                       use_amp=bool(args.amp), model_type=args.model_type, is_eval=True)
        if i == 0:
            pred_dict_vis = pred_dict
            batch_vis = batch

        loss_value, loss_details = loss_tuple  # criterion returns two values

        if isinstance(loss_value, tuple): # for evaluation datasets where <loss_value> are sample point clouds
            for m_name, m_val in loss_details.items():
                if "CD" in m_name:
                    loss_val = m_val[0] # chamfer <avg_val, count>
                    break
            metric_logger.update(loss=float(loss_val), **loss_details)
        else:
            metric_logger.update(loss=float(loss_value), **loss_details)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    aggs = [('avg', 'global_avg'), ('med', 'median')]
    results = {f'{k}_{tag}': getattr(meter, attr) for k, meter in metric_logger.meters.items() for tag, attr in aggs}


    if write_log:
        for name, val in results.items():
            wandb.log({f"{prefix}_{name}": val}, step=1000 * epoch)

        # save visualizations
        save_vis(batch_vis, pred_dict_vis, 1000 * epoch, args.n_save_intermediate, args.n_vis_layers, args.n_vis_pts3d, mode="ev")

    return results
