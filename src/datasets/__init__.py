from .utils.transforms import *
from .base.batched_sampler import BatchedRandomSampler  # noqa
from .objaverse import Objaverse
from .front3d import Front3D
from .gso import GSO
from .scrream import SCRREAM
from .scannetpp import ScanNetPP


def get_data_loader(dataset, batch_size, num_workers=8, shuffle=True, drop_last=True, pin_mem=True):
    import torch
    from src.utils.misc import get_world_size, get_rank

    # pytorch dataset
    if isinstance(dataset, str):
        dataset = eval(dataset)

    world_size = get_world_size()
    rank = get_rank()

    # we do distributed sampling with public APIs
    if torch.distributed.is_initialized():
        sampler = torch.utils.data.DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=shuffle, drop_last=drop_last
        )
    elif shuffle:
        sampler = torch.utils.data.RandomSampler(dataset)
    else:
        sampler = torch.utils.data.SequentialSampler(dataset)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_mem,
        drop_last=drop_last,
    )

    return data_loader
