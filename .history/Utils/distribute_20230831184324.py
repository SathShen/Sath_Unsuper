import os
import sys
import torch
import torch.distributed as dist




def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def init_distributed_mode(cfg):
    # launched with torch.distributed.launch
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        cfg.RANK = int(os.environ["RANK"])
        cfg.WORLD_SIZE = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    # launched with submitit on a slurm cluster
    elif 'SLURM_PROCID' in os.environ:
        cfg.RANK = int(os.environ['SLURM_PROCID'])
        local_rank = cfg.RANK % torch.cuda.device_count()
    # launched naively with `python main_dino.py`
    # we manually add MASTER_ADDR and MASTER_PORT to env variables
    elif torch.cuda.is_available():
        print('Will run the code on one GPU.')
        cfg.RANK, local_rank, cfg.WORLD_SIZE = 0, 0, 1
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29500'
    else:
        print('Does not support training without GPU.')
        sys.exit(1)

    dist.init_process_group(
        backend="nccl",
        world_size=cfg.WORLD_SIZE,
        rank=cfg.RANK,
    )

    torch.cuda.set_device(local_rank)
    dist.barrier()
    setup_for_distributed(cfg.RANK == 0)