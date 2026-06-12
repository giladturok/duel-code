#!/usr/bin/env python3
"""
Launch GSM8K experiments for DUEL perplexity vs downstream performance correlation.

Runs 8 experiments total:
- 4 generation experiments (GSM8K accuracy per strategy)
- 4 conditional likelihood experiments (DUEL perplexity per strategy)

Each job runs independently on a single GPU via SLURM (no sequential dependencies).

Usage:
    python scripts/launch_gsm8k_correlation.py              # Submit all
    python scripts/launch_gsm8k_correlation.py --filter gen  # Submit generation only
    python scripts/launch_gsm8k_correlation.py --filter cond # Submit conditional LL only
"""
import subprocess
import argparse

# ============================================================================
# CONFIG — tokens to unmask per step (baked into SLURM commands)
# ============================================================================
K = 2

# ============================================================================
# GENERATION EXPERIMENTS (downstream accuracy)
# ============================================================================

GENERATION_EXPERIMENTS = [
    # Block Greedy
    f"accelerate launch --num_processes=1 eval_block_permutation.py --tasks gsm8k --num_fewshot 4 --model llada_exact --batch_size 1 --model_args model_path='GSAI-ML/LLaDA-8B-Base',strategy=block_greedy,block_size=4,k={K},gen_length=256,use_wandb=True --output_path results --include_path tasks --log_samples",

    # Block Left-to-Right
    f"accelerate launch --num_processes=1 eval_block_permutation.py --tasks gsm8k --num_fewshot 4 --model llada_exact --batch_size 1 --model_args model_path='GSAI-ML/LLaDA-8B-Base',strategy=block_left_to_right,block_size=4,k={K},gen_length=256,use_wandb=True --output_path results --include_path tasks --log_samples",

    # Block Probability Margin
    f"accelerate launch --num_processes=1 eval_block_permutation.py --tasks gsm8k --num_fewshot 4 --model llada_exact --batch_size 1 --model_args model_path='GSAI-ML/LLaDA-8B-Base',strategy=block_probability_margin,block_size=4,k={K},gen_length=256,use_wandb=True --output_path results --include_path tasks --log_samples",

    # ELBO
    f"accelerate launch --num_processes=1 eval_llada_elbo.py --tasks gsm8k --num_fewshot 4 --model llada_elbo --batch_size 1 --model_args model_path='GSAI-ML/LLaDA-8B-Base',gen_length=256,k={K},remasking=random,use_wandb=True --output_path results --include_path tasks --log_samples",
]

# ============================================================================
# CONDITIONAL LIKELIHOOD EXPERIMENTS (DUEL perplexity)
# ============================================================================

CONDITIONAL_LL_EXPERIMENTS = [
    # Block Greedy (gen_length omitted — not used for conditional LL)
    f"accelerate launch --num_processes=1 eval_block_permutation.py --tasks gsm8k_cond_ll --num_fewshot 4 --model llada_exact --batch_size 1 --model_args model_path='GSAI-ML/LLaDA-8B-Base',strategy=block_greedy,block_size=4,k={K},use_wandb=True --output_path results --include_path tasks --log_samples",

    # Block Left-to-Right
    f"accelerate launch --num_processes=1 eval_block_permutation.py --tasks gsm8k_cond_ll --num_fewshot 4 --model llada_exact --batch_size 1 --model_args model_path='GSAI-ML/LLaDA-8B-Base',strategy=block_left_to_right,block_size=4,k={K},use_wandb=True --output_path results --include_path tasks --log_samples",

    # Block Probability Margin
    f"accelerate launch --num_processes=1 eval_block_permutation.py --tasks gsm8k_cond_ll --num_fewshot 4 --model llada_exact --batch_size 1 --model_args model_path='GSAI-ML/LLaDA-8B-Base',strategy=block_probability_margin,block_size=4,k={K},use_wandb=True --output_path results --include_path tasks --log_samples",

    # ELBO
    f"accelerate launch --num_processes=1 eval_llada_elbo.py --tasks gsm8k_cond_ll --num_fewshot 4 --model llada_elbo --batch_size 1 --model_args model_path='GSAI-ML/LLaDA-8B-Base',mc_num=128,k={K},use_wandb=True --output_path results --include_path tasks --log_samples",
]

EXPERIMENTS = GENERATION_EXPERIMENTS + CONDITIONAL_LL_EXPERIMENTS

# SLURM settings for single-GPU jobs
SBATCH_OVERRIDES = [
    "--gres=gpu:1",
    "--constraint=[a6000|a100|h100|h200]",
    "--partition=gpu",
    "--exclude=snavely-compute-02,abdelfattah-compute-02,seo-compute-02",
    "--time=72:00:00",
    "--mem=256G",
    "--cpus-per-task=4",
]


# ============================================================================
# Job Submission
# ============================================================================

def get_job_name(command):
    """Extract a job name from command."""
    parts = command.split()

    # Extract task
    task_idx = parts.index("--tasks") + 1 if "--tasks" in parts else -1
    task_name = parts[task_idx] if task_idx != -1 else "unknown"
    fewshot_idx = parts.index("--num_fewshot") + 1 if "--num_fewshot" in parts else -1
    fewshot = parts[fewshot_idx] if fewshot_idx != -1 else ""
    task = f"{task_name}_n{fewshot}" if fewshot != "" else task_name

    # Extract model name
    model_args = parts[parts.index("--model_args") + 1]
    model_name = model_args.split("model_path=")[1].split(",")[0].replace("'", "")
    model_name = model_name.replace("/", "_")

    # Determine method (exact or elbo)
    if "eval_block_permutation.py" in command or "eval_llada_exact.py" in command:
        method = "exact"
        strategy_args = [arg for arg in model_args.split(",") if arg.startswith("strategy=")]
        if strategy_args:
            strategy = strategy_args[0].split("=")[1]
            method += f"_{strategy}"
    elif "eval_llada_elbo.py" in command:
        method = "elbo"
        mc_args = [arg for arg in model_args.split(",") if arg.startswith("mc_num=")]
        if mc_args:
            mc_num = mc_args[0].split("=")[1]
            method += f"_mc{mc_num}"
    else:
        method = "unknown"

    return f"{task}/{model_name}/{method}"


def submit_job(command):
    """Submit command as independent SLURM job (single GPU)."""
    job_name = get_job_name(command)
    output_dir = f"outputs/{job_name}"

    parts = command.split()
    if "--output_path" in parts:
        idx = parts.index("--output_path")
        parts[idx + 1] = output_dir
    else:
        insert_idx = len(parts)
        parts.insert(insert_idx, "--output_path")
        parts.insert(insert_idx + 1, output_dir)

    modified_command = " ".join(parts)

    # Each job is independent — SLURM schedules them in parallel
    sbatch_cmd = [
        "sbatch",
        f"--job-name={job_name}",
        *SBATCH_OVERRIDES,
        "scripts/run_job.sbatch",
        modified_command
    ]

    result = subprocess.run(sbatch_cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"  {job_name}: {result.stdout.strip()}")
    else:
        print(f"  {job_name}: {result.stderr}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--filter", help="Only run experiments matching this string")
    args = parser.parse_args()

    experiments = EXPERIMENTS
    if args.filter:
        experiments = [cmd for cmd in experiments if args.filter.lower() in cmd.lower()]

    print(f"{'='*70}")
    print(f"Submitting {len(experiments)} GSM8K correlation experiments")
    print(f"  Each job: 1 GPU (A100/H100/H200), independent scheduling")
    print(f"{'='*70}")

    for cmd in experiments:
        submit_job(cmd)

    print(f"\n{'='*70}")
    print(f"Done! Check: squeue -u $USER")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
