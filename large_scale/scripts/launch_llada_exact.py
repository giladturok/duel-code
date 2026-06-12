#!/usr/bin/env python3
"""
Minimal experiment launcher - just a list of commands to run.

Usage:
    python launch_experiments.py              # Submit all
    python launch_experiments.py --filter gpqa  # Submit subset
"""
import subprocess
import argparse


# ============================================================================
# ALL EXPERIMENTS - Just add/edit commands here
# ============================================================================

MULTIPLE_CHOICE_EXPERIMENTS = [
    
    # GPQA
    # "accelerate launch eval_llada_exact.py --tasks gpqa_main_n_shot --num_fewshot 5 --model llada_exact --batch_size 16 --model_args model_path='GSAI-ML/LLaDA-8B-Base',cfg=0.0,is_check_greedy=False,strategy=block_greedy --output_path results --include_path tasks --log_samples",
    
    # TruthfulQA MC2
    "accelerate launch eval_llada_exact.py --tasks truthfulqa_mc2 --num_fewshot 0 --model llada_exact --batch_size 16 --model_args model_path='GSAI-ML/LLaDA-8B-Base',cfg=0.0,is_check_greedy=False,strategy=block_greedy --output_path results --include_path tasks --log_samples",
    
    # ARC Challenge
    "accelerate launch eval_llada_exact.py --tasks arc_challenge --num_fewshot 0 --model llada_exact --batch_size 16 --model_args model_path='GSAI-ML/LLaDA-8B-Base',cfg=0.0,is_check_greedy=False,strategy=block_greedy --output_path results --include_path tasks --log_samples",
    
    # Winogrande
    "accelerate launch eval_llada_exact.py --tasks winogrande --num_fewshot 5 --model llada_exact --batch_size 16 --model_args model_path='GSAI-ML/LLaDA-8B-Base',cfg=0.0,is_check_greedy=False,strategy=block_greedy --output_path results --include_path tasks --log_samples",
    
    # PIQA
    "accelerate launch eval_llada_exact.py --tasks piqa --num_fewshot 0 --model llada_exact --batch_size 16 --model_args model_path='GSAI-ML/LLaDA-8B-Base',cfg=0.0,is_check_greedy=False,strategy=block_greedy --output_path results --include_path tasks --log_samples",
    
    # HellaSwag
    "accelerate launch eval_llada_exact.py --tasks hellaswag --num_fewshot 0 --model llada_exact --batch_size 16 --model_args model_path='GSAI-ML/LLaDA-8B-Base',cfg=0.0,is_check_greedy=False,strategy=block_greedy --output_path results --include_path tasks --log_samples",
    
    # Storycloze
    "accelerate launch eval_llada_exact.py --tasks storycloze --num_fewshot 0 --model llada_exact --batch_size 16 --model_args model_path='GSAI-ML/LLaDA-8B-Base',cfg=0.0,is_check_greedy=False,strategy=block_greedy --output_path results --include_path tasks --log_samples",
    
    # MMLU
    "accelerate launch eval_llada_exact.py --tasks mmlu --num_fewshot 5 --model llada_exact --batch_size 16 --model_args model_path='GSAI-ML/LLaDA-8B-Base',cfg=0.0,is_check_greedy=False,strategy=block_greedy --output_path results --include_path tasks --log_samples",
    
    # CMMLU
    "accelerate launch eval_llada_exact.py --tasks cmmlu --num_fewshot 5 --model llada_exact --batch_size 16 --model_args model_path='GSAI-ML/LLaDA-8B-Base',cfg=0.0,is_check_greedy=False,strategy=block_greedy --output_path results --include_path tasks --log_samples",
    
    # C-Eval
    "accelerate launch eval_llada_exact.py --tasks ceval-valid --num_fewshot 5 --model llada_exact --batch_size 16 --model_args model_path='GSAI-ML/LLaDA-8B-Base',cfg=0.0,is_check_greedy=False,strategy=block_greedy --output_path results --include_path tasks --log_samples",
]

PERPLEXITY_EXPERIMENTS = [
    
    # Wikitext
    "accelerate launch eval_llada_exact.py --tasks wikitext --model llada_exact --batch_size 16 --model_args model_path='GSAI-ML/LLaDA-8B-Base',cfg=0.0,is_check_greedy=False,strategy=block_greedy --output_path results --include_path tasks --log_samples",
    
    # Openwebtext
    "accelerate launch eval_llada_exact.py --tasks openwebtext_perplexity --include_path tasks --model llada_exact --batch_size 16 --model_args model_path='GSAI-ML/LLaDA-8B-Base',cfg=0.0,is_check_greedy=False,strategy=block_greedy --output_path results --log_samples",
    
    # PennTreeBank
    "accelerate launch eval_llada_exact.py --tasks penntreebank_perplexity --include_path tasks --model llada_exact --batch_size 16 --model_args model_path='GSAI-ML/LLaDA-8B-Base',cfg=0.0,is_check_greedy=False,strategy=block_greedy --output_path results --log_samples",
    
    # Lambada
    "accelerate launch eval_llada_exact.py --tasks lambada_perplexity --include_path tasks --model llada_exact --batch_size 16 --model_args model_path='GSAI-ML/LLaDA-8B-Base',cfg=0.0,is_check_greedy=False,strategy=block_greedy --output_path results --log_samples",
    
    # Ag news
    "accelerate launch eval_llada_exact.py --tasks ag_news_perplexity --include_path tasks --model llada_exact --batch_size 16 --model_args model_path='GSAI-ML/LLaDA-8B-Base',cfg=0.0,is_check_greedy=False,strategy=block_greedy --output_path results --log_samples",
    
    # Arxiv
    "accelerate launch eval_llada_exact.py --tasks arxiv_perplexity --include_path tasks --model llada_exact --batch_size 16 --model_args model_path='GSAI-ML/LLaDA-8B-Base',cfg=0.0,is_check_greedy=False,strategy=block_greedy --output_path results --log_samples",
    
    # Pubmed
    "accelerate launch eval_llada_exact.py --tasks pubmed_perplexity --include_path tasks --model llada_exact --batch_size 16 --model_args model_path='GSAI-ML/LLaDA-8B-Base',cfg=0.0,is_check_greedy=False,strategy=block_greedy --output_path results --log_samples",
]

GENERATION_EXPERIMENTS = [
    # GSM8K
    # "accelerate launch eval_block_permutation.py --tasks gsm8k --num_fewshot 4 --model llada_exact --batch_size 32 --model_args model_path='GSAI-ML/LLaDA-8B-Base',strategy=block_greedy,block_size=4,gen_length=256 --output_path results --log_samples",
    "accelerate launch eval_block_permutation.py --tasks gsm8k --num_fewshot 4 --model llada_exact --batch_size 32 --model_args model_path='GSAI-ML/LLaDA-8B-Base',strategy=block_permutation,block_size=4,gen_length=256 --output_path results --log_samples",
    
    # HumanEval
    # "accelerate launch eval_block_permutation.py --tasks humaneval --num_fewshot 0 --model llada_exact --batch_size 32 --model_args model_path='GSAI-ML/LLaDA-8B-Base',strategy=block_greedy,block_size=4,gen_length=1024 --output_path results --log_samples  --confirm_run_unsafe_code",
    # "accelerate launch eval_block_permutation.py --tasks humaneval --num_fewshot 0 --model llada_exact --batch_size 32 --model_args model_path='GSAI-ML/LLaDA-8B-Base',strategy=block_permutation,block_size=4,gen_length=1024 --output_path results --log_samples --confirm_run_unsafe_code",
]

# EXPERIMENTS = MULTIPLE_CHOICE_EXPERIMENTS + PERPLEXITY_EXPERIMENTS
EXPERIMENTS=GENERATION_EXPERIMENTS

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
    model_name = model_name.replace("/", "_")  # escape the / if present
    
    # Determine method (exact or elbo)
    if "eval_llada_exact.py" in command:
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
        
    if "limit" in command:
        limit_idx = parts.index("--limit") + 1
        limit = parts[limit_idx]
        method += f"_lim{limit}"
    
    return f"{task}/{model_name}/{method}"


def submit_job(command):
    """Submit command as SLURM job with structured output path."""
    job_name = get_job_name(command)
    
    # Create timestamped output path: outputs/{timestamp}/{task}/{model}_{method}/results_{timestamp}.json
    output_dir = f"outputs/{job_name}"
    
    # Modify command to add/replace --output_path
    parts = command.split()
    if "--output_path" in parts:
        # Replace existing output_path
        idx = parts.index("--output_path")
        parts[idx + 1] = output_dir
    else:
        # Add output_path before --limit or --log_samples if present
        insert_idx = len(parts)
        for flag in ["--limit", "--log_samples"]:
            if flag in parts:
                insert_idx = min(insert_idx, parts.index(flag))
        parts.insert(insert_idx, "--output_path")
        parts.insert(insert_idx + 1, output_dir)
    
    modified_command = " ".join(parts)
    
    # SLURM command
    sbatch_cmd = [
        "sbatch",
        f"--job-name={job_name}",
        "scripts/run_job.sbatch",
        modified_command
    ]
    
    # Launch the job
    result = subprocess.run(sbatch_cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"✓ {job_name}: {result.stdout.strip()}")
    else:
        print(f"✗ {job_name}: {result.stderr}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--filter", help="Only run experiments matching this string")
    args = parser.parse_args()
    
    # Filter experiments
    experiments = EXPERIMENTS
    if args.filter:
        experiments = [cmd for cmd in experiments if args.filter.lower() in cmd.lower()]
    
    print(f"{'='*70}")
    print(f"Submitting {len(experiments)} experiments")
    print(f"{'='*70}")
    
    for cmd in experiments:
        submit_job(cmd)
    
    print(f"\n{'='*70}")
    print(f"Done! Check: squeue -u $USER")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()