import glob
import json
import multiprocessing
import subprocess
import argparse
import os
import random
import psutil


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_models_path", type=str, required=True,
                        help="Path to the GSO dataset root.")
    parser.add_argument("--object_path_file", type=str, required=True,
                        help="Path to JSON file with object name list.")
    parser.add_argument("--save_res_path", type=str, required=True)
    parser.add_argument("--ele_angles", type=int, nargs='+', default=[0, 30, 60])
    parser.add_argument("--azimuths_offset_angle", type=int, default=0)
    parser.add_argument("--num_images_per_ele", type=int, default=8)
    parser.add_argument("--ldi_layers", type=int, default=10)
    parser.add_argument("--camera_dist_low", type=float, default=1.0)
    parser.add_argument("--camera_dist_high", type=float, default=1.2)
    parser.add_argument("--render_first_n_scenes", type=int, default=-1)
    parser.add_argument("--render_timeout", type=int, default=2000)
    parser.add_argument("--online_sanity_check", type=int, default=0)
    parser.add_argument("--blender_path", type=str,
                        default="./blender-4.2.0-linux-x64",
                        help="Directory containing the blender executable")
    parser.add_argument("--workers_per_gpu", type=int, default=-1)
    parser.add_argument("--num_cpu_for_each_process", type=int, default=10)
    return parser.parse_args()


def worker(queue, count, gpu, cpu_group):
    psutil.Process(os.getpid()).cpu_affinity(cpu_group)

    while True:
        item = queue.get()
        if item is None:
            break

        object_path = item
        seq_name = item.split('/')[-3]
        view_path = os.path.join(args.save_res_path, seq_name)

        npy_files = glob.glob(os.path.join(view_path, "*.npy"))
        npz_files = glob.glob(os.path.join(view_path, "*.npz"))
        imgs_per_obj = args.num_images_per_ele * len(args.ele_angles)

        if os.path.exists(view_path) and (len(npy_files) + len(npz_files) == imgs_per_obj * 2):
            queue.task_done()
            print('========', item, 'already rendered', '========')
            continue

        img_script_dir = os.path.join(os.path.dirname(__file__), "gso/zero123_gso_script.py")
        ldi_script_dir = os.path.join(os.path.dirname(__file__), "ldi_render_per_object.py")
        rescaled_obj_dir = os.path.join(view_path, "res_normed.obj")
        angles = " ".join(map(str, args.ele_angles))

        command = (
            f"{args.blender_path}/blender -b -P {img_script_dir} -- "
            f"--object_path {object_path} --only_northern_hemisphere 1 "
            f"--num_images_per_ele {args.num_images_per_ele} --output_dir {view_path} "
            f"--ele_angles {angles} --camera_dist_low {args.camera_dist_low} "
            f"--camera_dist_high {args.camera_dist_high} "
            f"--azimuths_offset_angle {args.azimuths_offset_angle} && "
            f"python {ldi_script_dir} --object_path {rescaled_obj_dir} --camera_path {view_path} "
            f"--view_number {imgs_per_obj} --num_layers {args.ldi_layers} "
            f"--online_sanity_check {args.online_sanity_check} --dataset_type gso"
        )

        subprocess.run(
            ["bash", "-c", command],
            timeout=args.render_timeout,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        print(f"{seq_name}")

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

    allocated_cpus = sorted(list(os.sched_getaffinity(0)))
    cpu_groups = [
        allocated_cpus[i: i + args.num_cpu_for_each_process]
        for i in range(0, len(allocated_cpus), args.num_cpu_for_each_process)
    ]
    print(f'cpu_groups:{cpu_groups}')

    workers = []
    for worker_i, cpu_group in enumerate(cpu_groups):
        process = multiprocessing.Process(target=worker, args=(queue, count, worker_i, cpu_group))
        process.daemon = True
        process.start()
        workers.append(process)

    with open(args.object_path_file, "rt") as f:
        model_keys = json.load(f)

    models_to_render = model_keys if args.render_first_n_scenes == -1 else model_keys[:args.render_first_n_scenes]
    random.shuffle(models_to_render)

    relative_path = '{}/meshes/model.obj'
    for item in models_to_render:
        queue.put(os.path.join(args.input_models_path, relative_path.format(item)))

    queue.join()

    for _ in workers:
        queue.put(None)

    for process in workers:
        process.join()

    print("finished!!")
