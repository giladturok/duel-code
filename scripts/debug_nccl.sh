#!/bin/bash
#SBATCH -J debug_nccl
#SBATCH -o watch_folder/%x_%j.out
#SBATCH -e watch_folder/%x_%j.err
#SBATCH -N 1
#SBATCH --get-user-env
#SBATCH --mem=32G
#SBATCH -t 00:10:00
#SBATCH --partition=gpu
#SBATCH --constraint="[a5000|a6000|a100]"
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:4

export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

# Set MASTER_ADDR/PORT for torch.distributed rendezvous
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=29500

# Minimal NCCL test — no model, no data, just DDP init + one allreduce
srun python -u -c "
import os, torch, torch.distributed as dist

rank = int(os.environ.get('SLURM_PROCID', 0))
local_rank = int(os.environ.get('SLURM_LOCALID', 0))
world_size = int(os.environ.get('SLURM_NTASKS', 1))

print(f'[RANK {rank}] Starting on {os.environ.get(\"SLURMD_NODENAME\",\"?\")} GPU {local_rank}, MASTER_ADDR={os.environ.get(\"MASTER_ADDR\",\"?\")}, world_size={world_size}', flush=True)

torch.cuda.set_device(local_rank)

dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
print(f'[RANK {rank}] init_process_group done', flush=True)

# Simple allreduce test
tensor = torch.ones(1).cuda()
dist.all_reduce(tensor)
print(f'[RANK {rank}] all_reduce result: {tensor.item()} (expected {world_size})', flush=True)

dist.barrier()
print(f'[RANK {rank}] barrier passed — NCCL is working', flush=True)

dist.destroy_process_group()
print(f'[RANK {rank}] DONE', flush=True)
"
