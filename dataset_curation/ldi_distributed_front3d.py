'''
Render images and LDIs for each 3D-FRONT room.

The loaded mesh is the whole scene but split into a single room by <room_id>.
Both image and LDI are rendered from the selected room's mesh.
'''

import glob
import json
import multiprocessing
import subprocess
import time
import argparse
import os
import random
import psutil


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--object_path_file", type=str, required=True,
                        help="Path to JSON file with room list (house_name room_id entries)")
    parser.add_argument("--save_res_path", type=str, required=True)
    parser.add_argument("--cam_len_low", type=int, required=True)
    parser.add_argument("--cam_len_high", type=int, required=True)
    parser.add_argument("--num_renders", type=int, default=12)
    parser.add_argument("--ldi_layers", type=int, default=15)
    parser.add_argument("--render_first_n_scenes", type=int, default=-1)
    parser.add_argument("--render_timeout", type=int, default=1200)
    parser.add_argument("--online_sanity_check", type=int, default=0)
    parser.add_argument("--workers_per_gpu", type=int, default=-1)
    parser.add_argument("--num_cpu_for_each_process", type=int, default=10)
    parser.add_argument("--front", type=str,
                        default="./data/3d_front/3D-FRONT")
    parser.add_argument("--future_folder", type=str,
                        default="./data/3d_front/3D-FUTURE-model")
    parser.add_argument("--front_3D_texture_path", type=str,
                        default="./data/3d_front/3D-FRONT-texture")
    parser.add_argument("--cc_material_path", type=str,
                        default="./data/3d_front/cc_texture")
    return parser.parse_args()


def worker(queue, count, gpu, cpu_group):
    psutil.Process(os.getpid()).cpu_affinity(cpu_group)

    while True:
        item = queue.get()
        if item is None:
            break

        house_name, room_id = item.split(' ')
        house_name = house_name.split(".")[0]
        room_id = int(room_id)

        house_path = os.path.join(args.front, house_name + ".json")
        view_path = os.path.join(args.save_res_path, "{}_{}".format(house_name, room_id))
        obj_path = os.path.join(view_path, "res.obj")

        npy_files = glob.glob(os.path.join(view_path, "*.npy"))
        npz_files = glob.glob(os.path.join(view_path, "*.npz"))

        if os.path.exists(view_path) and (len(npy_files) + len(npz_files) == 2 * args.num_renders):
            queue.task_done()
            print('========', item, 'already rendered', '========')
            continue

        img_script_dir = os.path.join(os.path.dirname(__file__), "front3d/img_render_front3d_per_room.py")
        ldi_script_dir = os.path.join(os.path.dirname(__file__), "ldi_render_per_object.py")

        assert args.cam_len_low <= args.cam_len_high
        rand_cam_len = random.randint(args.cam_len_low, args.cam_len_high)

        command = (
            f"blenderproc run {img_script_dir} --front {house_path} --room_id {room_id} "
            f"--cam_lens_range {rand_cam_len} {rand_cam_len} --future_folder {args.future_folder} "
            f"--front_3D_texture_path {args.front_3D_texture_path} --cc_material_path {args.cc_material_path} "
            f"--num_rendering {args.num_renders} --output_dir {view_path} && "
            f"python {ldi_script_dir} --object_path {obj_path} --camera_path {view_path} "
            f"--view_number {args.num_renders} --num_layers {args.ldi_layers} "
            f"--online_sanity_check {args.online_sanity_check} --dataset_type front3d"
        )

        subprocess.run(
            ["bash", "-c", command],
            timeout=args.render_timeout,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        print(f"{house_name}")

        with count.get_lock():
            count.value += 1

        queue.task_done()


def get_allocated_cpus():
    return sorted(list(os.sched_getaffinity(0)))


if __name__ == "__main__":
    args = parse_args()
    start_t = time.time()

    queue = multiprocessing.JoinableQueue()
    count = multiprocessing.Value("i", 0)

    if args.workers_per_gpu == -1:
        args.workers_per_gpu = int(os.environ.get('SLURM_CPUS_PER_TASK', 4))
    print(f'cpus:{args.workers_per_gpu}')

    allocated_cpus = get_allocated_cpus()
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

    with open(args.object_path_file, "r") as f:
        model_keys = json.load(f)
        model_keys = [obj_room_name for obj_room_name, _ in model_keys.items()]

    models_to_render = model_keys if args.render_first_n_scenes == -1 else model_keys[:args.render_first_n_scenes]
    random.shuffle(models_to_render)

    for item in models_to_render:
        queue.put(item)

    queue.join()

    for _ in workers:
        queue.put(None)

    for process in workers:
        process.join()

    end_t = time.time()
    print("time elapsed: {}".format(end_t - start_t))
    print("finished!!")
