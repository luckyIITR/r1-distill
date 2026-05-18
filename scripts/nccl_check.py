"""30-second NCCL sanity check. Run with:
    torchrun --nproc_per_node=2 scripts/nccl_check.py
"""
import os
import time
import torch
import torch.distributed as dist


def main():
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", timeout=__import__("datetime").timedelta(seconds=60))

    print(f"[rank {rank}] init complete, world={world}, device={torch.cuda.current_device()}")

    # All-reduce a small tensor
    t = torch.ones(8, device=f"cuda:{local_rank}") * (rank + 1)
    t0 = time.time()
    dist.all_reduce(t)
    torch.cuda.synchronize()
    print(f"[rank {rank}] all_reduce small: result={t[0].item()}, took {time.time()-t0:.3f}s")

    # All-reduce a 1GB tensor (realistic for ZeRO bucket sizes)
    big = torch.ones(int(2.5e8), device=f"cuda:{local_rank}", dtype=torch.bfloat16)
    torch.cuda.synchronize()
    t0 = time.time()
    dist.all_reduce(big)
    torch.cuda.synchronize()
    print(f"[rank {rank}] all_reduce 500MB: took {time.time()-t0:.3f}s")

    dist.barrier()
    if rank == 0:
        print("\n✅ NCCL sanity check passed")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()