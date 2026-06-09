import glob
import json
import multiprocessing
import subprocess
import argparse
import os
import random
import gzip


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_models_path", type=str, required=True,
                        help="Path to Objaverse dataset root.")
    parser.add_argument("--object_path_file", type=str, required=True,
                        help="Path to (gzipped) JSON file with object paths.")
    parser.add_argument("--save_res_path", type=str, required=True)
    parser.add_argument("--object_bundle_range", type=str, nargs=2, default=None,
                        help="Range of bundle folders to process (e.g. 000 050).")
    parser.add_argument("--num_renders", type=int, default=12)
    parser.add_argument("--ldi_layers", type=int, default=15)
    parser.add_argument("--only_northern_hemisphere", type=int, default=1)
    parser.add_argument("--render_first_n_scenes", type=int, default=-1)
    parser.add_argument("--render_timeout", type=int, default=600)
    parser.add_argument("--online_sanity_check", type=int, default=0)
    parser.add_argument("--blender_path", type=str,
                        default="./blender-4.2.0-linux-x64",
                        help="Directory containing the blender executable")
    parser.add_argument("--workers_per_gpu", type=int, default=-1)
    return parser.parse_args()


def worker(queue, count, gpu):
    while True:
        item = queue.get()
        if item is None:
            break

        glb_name = item.split('/')[-1][:-4]
        seq_name = item.split('/')[-2]
        view_path = os.path.join(args.save_res_path, seq_name, glb_name)

        png_files = glob.glob(os.path.join(view_path, "*.png"))
        npy_files = glob.glob(os.path.join(view_path, "*.npy"))
        npz_files = glob.glob(os.path.join(view_path, "*.npz"))

        # skip if images, camera poses, and LDIs are all rendered
        if os.path.exists(view_path) and (len(png_files) + len(npy_files) + len(npz_files) == 3 * args.num_renders):
            queue.task_done()
            print('========', item, 'already rendered', '========')
            continue

        os.makedirs(view_path, exist_ok=True)

        img_script_dir = os.path.join(os.path.dirname(__file__), "objaverse/zero123_blender_script.py")
        ldi_script_dir = os.path.join(os.path.dirname(__file__), "ldi_render_per_object.py")

        command = (
            f"{args.blender_path}/blender -b -P {img_script_dir} -- "
            f"--object_path {item} --only_northern_hemisphere {args.only_northern_hemisphere} "
            f"--num_images {args.num_renders} --output_dir {args.save_res_path} && "
            f"python {ldi_script_dir} --object_path {item} --camera_path {view_path} "
            f"--view_number {args.num_renders} --num_layers {args.ldi_layers} "
            f"--online_sanity_check {args.online_sanity_check} --dataset_type objaverse"
        )

        subprocess.run(
            ["bash", "-c", command],
            timeout=args.render_timeout,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        print(f"{seq_name}/{glb_name}")

        with count.get_lock():
            count.value += 1

        queue.task_done()


if __name__ == "__main__":
    args = parse_args()

    queue = multiprocessing.JoinableQueue()
    count = multiprocessing.Value("i", 0)

    if args.workers_per_gpu == -1:
        args.workers_per_gpu = int(os.environ.get('SLURM_CPUS_PER_TASK', 4))
    print(f'cpus:{args.workers_per_gpu}')

    workers = []
    for worker_i in range(args.workers_per_gpu):
        process = multiprocessing.Process(target=worker, args=(queue, count, 0))
        process.daemon = True
        process.start()
        workers.append(process)

    def load_json(path):
        opener = gzip.open if path.endswith(".gz") else open
        with opener(path, "rt", encoding="utf-8") as f:
            return json.load(f)

    if args.object_bundle_range is None:
        model_paths = load_json(args.object_path_file)
    else:
        model_paths = {}
        for bund_i in range(int(args.object_bundle_range[0]), int(args.object_bundle_range[1])):
            model_paths.update(
                load_json(os.path.join(args.object_path_file, "000-{:03d}.json.gz".format(bund_i)))
            )

    model_keys = list(model_paths.keys())
    models_to_render = model_keys if args.render_first_n_scenes == -1 else model_keys[:args.render_first_n_scenes]
    random.shuffle(models_to_render)

    for item in models_to_render:
        queue.put(os.path.join(args.input_models_path, model_paths[item]))

    queue.join()

    for _ in workers:
        queue.put(None)

    for process in workers:
        process.join()

    print("finished!!")
