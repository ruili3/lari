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
                        help="Path to the ScanNet++ dataset root.")
    parser.add_argument("--object_path_file", type=str, required=True,
                        help="Path to JSON file with scene/image list.")
    parser.add_argument("--ldi_layers", type=int, default=10)
    parser.add_argument("--render_first_n_scenes", type=int, default=-1)
    parser.add_argument("--online_sanity_check", type=int, default=0)
    parser.add_argument("--workers_per_gpu", type=int, default=-1)
    parser.add_argument("--num_cpu_for_each_process", type=int, default=10)
    parser.add_argument("--point_priority_thres", type=int, default=None)
    parser.add_argument("--render_timeout", type=int, default=None,
                        help="Timeout in seconds for each render subprocess. None means no timeout.")
    return parser.parse_args()


def worker(queue, count, gpu, cpu_group):
    psutil.Process(os.getpid()).cpu_affinity(cpu_group)

    while True:
        item = queue.get()
        if item is None:
            break

        # item format: "{scene_abs_path} {file_id}"
        scene_abs_path, file_id = item.split(" ")

        ldi_save_path = os.path.join(scene_abs_path, "dslr/ldi")

        png_file = os.path.join(ldi_save_path, "{}_ldi.png".format(file_id))
        npz_file = os.path.join(ldi_save_path, "{}_ldi.npz".format(file_id))
        if os.path.exists(png_file) and os.path.exists(npz_file):
            queue.task_done()
            print('========', item, 'already rendered', '========')
            continue

        ldi_script_dir = os.path.join(os.path.dirname(__file__), "ldi_render_per_object.py")

        command = (
            f"python {ldi_script_dir} --object_path {scene_abs_path} --camera_path {file_id} "
            f"--num_layers {args.ldi_layers} --online_sanity_check {args.online_sanity_check} "
            f"--dataset_type scannetpp --point_priority_thres {args.point_priority_thres}"
        )

        subprocess.run(
            ["bash", "-c", command],
            timeout=args.render_timeout,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        print(f"{item}")

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

    for item in models_to_render:
        # item format: "00777c41d4 DSC00920"
        scene_name, file_id = item.split(" ")
        scene_abs_path = os.path.join(args.input_models_path, scene_name)
        queue.put(f"{scene_abs_path} {file_id}")

    queue.join()

    for _ in workers:
        queue.put(None)

    for process in workers:
        process.join()

    print("finished!!")
