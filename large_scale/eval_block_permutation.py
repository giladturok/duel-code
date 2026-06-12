"""
Unified evaluation wrapper for LLaDA models with exact likelihood strategies.

Supports:
- block_greedy: Low confidence remasking (like LLaDA generate.py)
- block_permutation: Exhaustive permutation search
- block_left_to_right: Left-to-right within blocks
- greedy_confidence: Standard greedy decoding

Usage:
    # Single GPU - Block Greedy
    python eval_llada_exact.py \
        --tasks arc_challenge \
        --num_fewshot 25 \
        --model llada_exact \
        --batch_size 4 \
        --model_args model_path='GSAI-ML/LLaDA-8B-Base',strategy=block_greedy,block_size=32
    
    # Multi-GPU - Block Greedy
    accelerate launch eval_llada_exact.py \
        --tasks arc_challenge \
        --num_fewshot 25 \
        --model llada_exact \
        --batch_size 4 \
        --model_args model_path='GSAI-ML/LLaDA-8B-Base',strategy=block_greedy,block_size=32
    
    # Block Permutation (must use batch_size=1)
    accelerate launch eval_llada_exact.py \
        --tasks arc_challenge \
        --num_fewshot 25 \
        --model llada_exact \
        --batch_size 1 \
        --model_args model_path='GSAI-ML/LLaDA-8B-Base',strategy=block_permutation,block_size=4
"""
import os
os.environ["HF_ALLOW_CODE_EVAL"] = "1"
import warnings
warnings.filterwarnings('ignore', category=Warning, module='dill')

import math
import re
import sys
import accelerate
import torch
import torch.nn.functional as F
from pathlib import Path
from datasets import Dataset
from lm_eval.__main__ import cli_evaluate
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import wandb
import os

# Import our strategies and compute functions
from exact_likelihood.compute import (
    compute_deterministic_loglikelihood,
    compute_block_permutation_loglikelihood,
    generate_with_strategy,
)
from exact_likelihood.strategies import (
    GreedyConfidenceStrategy,
    BlockGreedyConfidenceStrategy,
    BlockLeftToRightStrategy,
    BlockProbabilityMarginStrategy,
    BlockPermutationStrategy,
)

# datasets.config.HF_DATASETS_TRUST_REMOTE_CODE = True

@register_model("llada_exact")
class LLaDAExactEvalHarness(LM):
    """
    LLaDA evaluation with exact likelihood computation using various strategies.
    
    Strategies:
    - block_greedy: Greedy confidence within blocks (like LLaDA's low_confidence)
    - block_permutation: Exhaustive permutation search (expensive, batch_size must be 1)
    - block_left_to_right: Left-to-right within blocks
    - greedy_confidence: Standard greedy (no blocks)
    """
    
    def __init__(
        self,
        model_path='GSAI-ML/LLaDA-8B-Base',
        mask_token_id=126336,
        max_length=1028,
        batch_size=4,
        strategy='block_greedy',  # or 'block_permutation', 'block_left_to_right', 'greedy_confidence'
        block_size=32,
        k=1,  # tokens to unmask per step
        gen_length=256,  # only used for generation tasks, not conditional likelihood
        device="cuda",
        use_wandb=True,
        wandb_project="llada-exact-likelihood",
        wandb_entity=None,
        wandb_run_name=None,
        wandb_tags=None,
        **kwargs,
    ):
        """
        Args:
            model_path: Path to LLaDA model
            mask_token_id: Token ID for [MASK] (126336 for LLaDA)
            batch_size: Batch size (must be 1 for block_permutation)
            strategy: Decoding strategy to use
            block_size: Size of blocks (32 for block_greedy, 4 for block_permutation)
            gen_length: Number of tokens to generate
            use_wandb: Whether to log to Weights & Biases
            wandb_project: W&B project name
            wandb_entity: W&B entity (team/user)
            wandb_run_name: W&B run name (auto-generated if None)
            wandb_tags: List of tags for W&B run
        """
        super().__init__()

        # Cast numeric args (lm-eval passes everything as strings from --model_args)
        batch_size = int(batch_size)
        block_size = int(block_size)
        k = int(k)
        max_length = int(max_length)
        gen_length = int(gen_length)
        mask_token_id = int(mask_token_id)
        if isinstance(use_wandb, str):
            use_wandb = use_wandb.lower() not in ('false', '0', 'no')

        # Validate strategy and batch_size
        self.strategy_name = strategy
        if strategy == 'block_permutation' and batch_size != 1:
            warnings.warn(
                f"block_permutation requires batch_size=1 (batching is internal). "
                f"Setting batch_size to 1."
            )
            batch_size = 1
        
        # Setup accelerate for multi-GPU
        accelerator = accelerate.Accelerator()
        self.accelerator = accelerator if accelerator.num_processes > 1 else None
        
        # Load model with proper device placement
        model_kwargs = {}
        if self.accelerator is not None:
            model_kwargs['device_map'] = {'': f'{self.accelerator.device}'}
            device = self.accelerator.device
        
        self.model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            **model_kwargs
        )
        self.model.eval()
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
        
        # Setup device and distributed info
        if self.accelerator is not None:
            self.model = self.accelerator.prepare(self.model)
            self.device = torch.device(f'{self.accelerator.device}')
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self.model = self.model.to(device)
            self.device = torch.device(device)
            self._rank = 0
            self._world_size = 1
        
        self.mask_token_id = mask_token_id
        self.max_length = max_length
        self.batch_size = batch_size
        self.block_size = block_size
        self.gen_length = gen_length
        
        # Create strategy instance
        self.strategy = self._create_strategy(strategy, block_size, k)

        # Get k (tokens per step) from strategy
        self.k = getattr(self.strategy, 'k', 1)

        # Initialize Weights & Biases (only on rank 0)
        self.use_wandb = use_wandb
        if self.use_wandb and self._rank == 0:
            # Parse wandb tags
            # Extract task name from CLI args for distinguishing runs
            task_name = "unknown"
            if "--tasks" in sys.argv:
                task_name = sys.argv[sys.argv.index("--tasks") + 1]

            # Infer task type from task name
            task_type = "likelihood" if "cond_ll" in task_name else "generation"

            # Parse limit from CLI args
            limit = None
            if "--limit" in sys.argv:
                limit = int(sys.argv[sys.argv.index("--limit") + 1])

            if wandb_tags is None:
                wandb_tags = [strategy, f"block_size_{block_size}", f"k_{self.k}", task_type]
                if limit is not None:
                    wandb_tags.append(f"limit_{limit}")
            elif isinstance(wandb_tags, str):
                wandb_tags = [t.strip() for t in wandb_tags.split(',')]

            # Auto-generate run name if not provided
            if wandb_run_name is None:
                limit_suffix = f"_lim{limit}" if limit is not None else ""
                wandb_run_name = f"{strategy}_bs{block_size}_k{self.k}_batch{batch_size}_{task_name}{limit_suffix}"

            # Initialize wandb
            wandb.init(
                project=wandb_project,
                entity=wandb_entity,
                name=wandb_run_name,
                tags=wandb_tags,
                config={
                    "model_path": model_path,
                    "strategy": strategy,
                    "block_size": block_size,
                    "k": self.k,
                    "batch_size": batch_size,
                    "gen_length": gen_length,
                    "max_length": max_length,
                    "mask_token_id": mask_token_id,
                    "num_gpus": self._world_size,
                    "limit": limit,
                }
            )
            
            print(f"[Rank {self._rank}] Initialized W&B: {wandb.run.name}")
        
        # Iteration counter for logging
        self.iteration = 0
        
        print(f"[Rank {self._rank}/{self._world_size}] Initialized LLaDA-Exact with:")
        print(f"  Strategy: {strategy}")
        print(f"  Block size: {block_size}")
        print(f"  Batch size: {batch_size}")
        print(f"  W&B logging: {use_wandb and self._rank == 0}")
        if strategy == 'block_permutation':
            print(f"  Permutations per block: {math.factorial(min(block_size, 5))}")
    
    def _create_strategy(self, strategy_name, block_size, k=1):
        """Create strategy instance based on name."""
        if strategy_name == 'block_greedy':
            return BlockGreedyConfidenceStrategy(block_size=block_size, k=k)
        elif strategy_name == 'block_permutation':
            return BlockPermutationStrategy(block_size=block_size)
        elif strategy_name == 'block_left_to_right':
            return BlockLeftToRightStrategy(block_size=block_size, k=k)
        elif strategy_name == 'block_probability_margin':
            return BlockProbabilityMarginStrategy(block_size=block_size, k=k)
        elif strategy_name == 'greedy_confidence':
            return GreedyConfidenceStrategy(k=k)
        else:
            raise ValueError(
                f"Unknown strategy: {strategy_name}. "
                f"Must be one of: block_greedy, block_permutation, "
                f"block_left_to_right, block_probability_margin, greedy_confidence"
            )
    
    def __del__(self):
        """Cleanup: finish wandb run if active."""
        if getattr(self, 'use_wandb', False) and getattr(self, '_rank', 1) == 0 and wandb.run is not None:
            wandb.finish()
    
    @property
    def rank(self):
        return self._rank
    
    @property
    def world_size(self):
        return self._world_size
    
    def _encode_pair(self, context, continuation):
        """Encode context + continuation, handling whitespace."""
        n_spaces = len(context) - len(context.rstrip())
        if n_spaces > 0:
            continuation = context[-n_spaces:] + continuation
            context = context[:-n_spaces]
        
        whole_enc = self.tokenizer(context + continuation)["input_ids"]
        context_enc = self.tokenizer(context)["input_ids"]
        
        context_enc_len = len(context_enc)
        continuation_enc = whole_enc[context_enc_len:]
        
        return context_enc, continuation_enc
    
    @torch.no_grad()
    def loglikelihood(self, requests):
        """
        Compute log-likelihood using the specified strategy.
        
        For multi-GPU: Each rank processes a subset of requests.
        """
        def _tokenize(e):
            prefix, target = self._encode_pair(e["prefix_text"], e["target_text"])
            return {
                "prefix": prefix,
                "target": target,
            }

        ds = [{"prefix_text": req.args[0], "target_text": req.args[1]} for req in requests]
        ds = Dataset.from_list(ds)
        ds = ds.map(function=_tokenize)
        ds = ds.with_format("torch")
        
        # Handle truncation
        max_seq_length = max([len(x["prefix"]) + len(x["target"]) for x in ds])
        if max_seq_length > self.max_length:
            ds = ds.map(lambda x: {
                "prefix": x["prefix"][max(0, len(x["prefix"]) - (self.max_length - len(x["target"]))):],
                "target": x["target"][:self.max_length]
            })
        
        # For block_permutation, ensure answer length is divisible by block_size
        if self.strategy_name == 'block_permutation':
            def _pad_to_block(x):
                answer_len = len(x["target"])
                if answer_len % self.block_size != 0:
                    pad_len = self.block_size - (answer_len % self.block_size)
                    x["target"] = torch.cat([
                        x["target"],
                        torch.full((pad_len,), self.tokenizer.pad_token_id, dtype=torch.long)
                    ])
                return x
            ds = ds.map(_pad_to_block)
        
        out = []
        
        # Process in batches
        desc = f"Computing likelihood (Rank {self._rank})" if self._world_size > 1 else "Computing likelihood"
        
        for i in tqdm(range(0, len(ds), self.batch_size), desc=desc, disable=self._rank != 0):
            batch = [ds[j] for j in range(i, min(i + self.batch_size, len(ds)))]

            # Handle variable-length batching
            max_len = max([len(elem["prefix"]) + len(elem["target"]) for elem in batch])

            # Pad sequences in batch
            tokens_list = []
            prompt_lens_list = []
            lengths_list = []

            for elem in batch:
                tokens = torch.cat([elem["prefix"], elem["target"]])
                prompt_len = len(elem["prefix"])
                length = len(tokens)
                
                # Pad to max_len
                if length < max_len:
                    tokens = torch.cat([
                        tokens,
                        torch.full((max_len - length,), self.tokenizer.pad_token_id, dtype=torch.long)
                    ])
                
                tokens_list.append(tokens)
                prompt_lens_list.append(prompt_len)
                lengths_list.append(length)
            
            # Stack into batch tensors
            tokens = torch.stack(tokens_list).to(self.device)
            prompt_lens = torch.tensor(prompt_lens_list, device=self.device)
            lengths = torch.tensor(lengths_list, device=self.device)
            
            # Compute likelihood based on strategy
            if self.strategy_name == 'block_permutation':
                ll = compute_block_permutation_loglikelihood(
                    model=self.model,
                    x=tokens,
                    prompt_lens=prompt_lens,
                    lengths=lengths,
                    mask_token_id=self.mask_token_id,
                    block_size=self.block_size,
                )
            else:
                ll = compute_deterministic_loglikelihood(
                    model=self.model,
                    x=tokens,
                    prompt_lens=prompt_lens,
                    lengths=lengths,
                    mask_token_id=self.mask_token_id,
                    strategy=self.strategy,
                )
            
            # Convert to list of (ll, is_greedy) tuples
            for j, ll_val in enumerate(ll):
                out.append((ll_val.item(), False))

                # Per-example W&B logging (rank 0 only)
                if self.use_wandb and self._rank == 0:
                    example_idx = i + j
                    answer_len = (lengths[j] - prompt_lens[j]).item()

                    # Compute perplexity
                    ppl = torch.exp(-ll_val / answer_len)

                    log_data = {
                        "iteration": self.iteration,
                        "example_idx": example_idx,
                        "log_likelihood": ll_val.item(),
                        "perplexity": ppl.item(),
                        "answer_length": answer_len,
                        "prompt_length": prompt_lens[j].item(),
                        "total_length": lengths[j].item(),
                        "nll_per_token": -ll_val.item() / answer_len,
                    }

                    # Log question/answer text if available
                    elem = batch[j]
                    if "prefix_text" in elem:
                        log_data["question"] = elem["prefix_text"]
                    if "target_text" in elem:
                        log_data["answer"] = elem["target_text"]

                    wandb.log(log_data)
                    self.iteration += 1
        
        # Synchronize across GPUs if using distributed
        if self.accelerator is not None:
            self.accelerator.wait_for_everyone()
        
        # Log summary statistics (rank 0 only)
        if self.use_wandb and self._rank == 0 and len(out) > 0:
            lls = [x[0] for x in out]
            wandb.log({
                "summary/mean_ll": sum(lls) / len(lls),
                "summary/num_examples": len(lls),
            })
        
        return out
    
    @torch.no_grad()
    def loglikelihood_rolling(self, requests):
        """
        Compute rolling log-likelihood for perplexity evaluation.
        """
        if self._rank == 0:
            print(f"Total documents: {len(requests)}")
        
        # Tokenize documents
        all_tokens = [
            self.tokenizer(req.args[0])["input_ids"] 
            for req in tqdm(requests, desc="Tokenizing...", disable=self._rank != 0)
        ]
        
        out = []
        desc = f"Computing rolling likelihood (Rank {self._rank})" if self._world_size > 1 else "Computing rolling likelihood"
        
        for doc_idx, tokens in enumerate(tqdm(all_tokens, desc=desc, disable=self._rank != 0)):
            doc_ll = 0.0
            num_chunks = 0
            
            # Process in chunks
            for start_idx in range(0, len(tokens), self.max_length):
                chunk = tokens[start_idx:start_idx + self.max_length]
                
                # For block strategies, pad to block_size multiple
                if 'block' in self.strategy_name:
                    if len(chunk) % self.block_size != 0:
                        pad_len = self.block_size - (len(chunk) % self.block_size)
                        chunk = chunk + [self.tokenizer.pad_token_id] * pad_len
                
                chunk_tokens = torch.tensor(chunk, dtype=torch.long).unsqueeze(0).to(self.device)
                prompt_lens = torch.tensor([0], device=self.device)
                lengths = torch.tensor([len(chunk)], device=self.device)
                
                if self.strategy_name == 'block_permutation':
                    ll = compute_block_permutation_loglikelihood(
                        model=self.model,
                        x=chunk_tokens,
                        prompt_lens=prompt_lens,
                        lengths=lengths,
                        mask_token_id=self.mask_token_id,
                        block_size=self.block_size,
                    )
                else:
                    ll = compute_deterministic_loglikelihood(
                        model=self.model,
                        x=chunk_tokens,
                        prompt_lens=prompt_lens,
                        lengths=lengths,
                        mask_token_id=self.mask_token_id,
                        strategy=self.strategy,
                    )
                
                chunk_ll = ll.item()
                doc_ll += chunk_ll
                num_chunks += 1
                
                # Per-chunk W&B logging (rank 0 only)
                if self.use_wandb and self._rank == 0:
                    chunk_ppl = torch.exp(-ll / len(chunk))
                    
                    wandb.log({
                        "rolling/iteration": self.iteration,
                        "rolling/doc_idx": doc_idx,
                        "rolling/chunk_idx": num_chunks - 1,
                        "rolling/chunk_ll": chunk_ll,
                        "rolling/chunk_ppl": chunk_ppl.item(),
                        "rolling/chunk_length": len(chunk),
                    })
                    
                    self.iteration += 1
            
            out.append(doc_ll)
            
            # Per-document W&B logging (rank 0 only)
            if self.use_wandb and self._rank == 0:
                doc_ppl = torch.exp(torch.tensor(-doc_ll / len(tokens)))
                
                wandb.log({
                    "rolling/doc_idx": doc_idx,
                    "rolling/doc_ll": doc_ll,
                    "rolling/doc_ppl": doc_ppl.item(),
                    "rolling/doc_length": len(tokens),
                    "rolling/num_chunks": num_chunks,
                })
        
        if self.accelerator is not None:
            self.accelerator.wait_for_everyone()
        
        # Log summary statistics (rank 0 only)
        if self.use_wandb and self._rank == 0 and len(out) > 0:
            total_ll = sum(out)
            total_tokens = sum(len(tokens) for tokens in all_tokens)
            overall_ppl = math.exp(-total_ll / total_tokens)
            
            wandb.log({
                "rolling_summary/total_ll": total_ll,
                "rolling_summary/total_tokens": total_tokens,
                "rolling_summary/overall_ppl": overall_ppl,
                "rolling_summary/num_documents": len(out),
            })
        
        return out
    
    @torch.no_grad()
    def generate_until(self, requests: list[Instance]):
        """
        Generate text using the specified strategy.
        """
        def _tokenize(e):
            return {
                "question": self.tokenizer(e["question"])["input_ids"],
                "until": e["until"],
                "doc_answer": e["doc_answer"],
            }
        
        # Extract ground truth answer number from doc if available (e.g., GSM8K)
        ANSWER_PATTERN = re.compile(r"####\s*(\-?[0-9\.\,]+)")

        def _extract_answer_number(text):
            """Extract the number after #### from GSM8K-style answers."""
            match = ANSWER_PATTERN.search(text)
            if match:
                return match.group(1).replace(",", "")
            return None

        ds = [{"question": req.args[0], "until": req.args[1]['until'],
               "doc_answer": req.doc.get("answer", "") if hasattr(req, "doc") and req.doc else ""}
              for req in requests]
        ds = Dataset.from_list(ds)
        ds = ds.map(_tokenize)
        ds = ds.with_format("torch")
        
        # Reset strategy state if needed
        if hasattr(self.strategy, 'reset_state'):
            self.strategy.reset_state(1, self.device)  # batch_size=1 for generation
        
        out = []
        num_correct = 0
        num_scored = 0
        desc = f"Generating (Rank {self._rank})" if self._world_size > 1 else "Generating"

        for elem_idx, elem in enumerate(tqdm(ds, desc=desc, disable=self._rank != 0)):
            prompt = elem["question"].unsqueeze(0).to(self.device)
            stop_tokens = elem["until"]
            
            # Ensure gen_length is compatible with block_size
            gen_length = self.gen_length
            if 'block' in self.strategy_name and gen_length % self.block_size != 0:
                gen_length = ((gen_length // self.block_size) + 1) * self.block_size
            
            # Generate using strategy
            generated = generate_with_strategy(
                model=self.model,
                prompt=prompt,
                gen_length=gen_length,
                strategy=self.strategy,
                mask_token_id=self.mask_token_id,
                block_size=self.block_size if 'block' in self.strategy_name else None,
            )
            
            # Decode and apply stop sequences
            generated_text = self.tokenizer.decode(
                generated[0][prompt.shape[1]:], skip_special_tokens=False
            )
            
            # Track which stop token was used
            stop_token_used = None
            for stop_seq in stop_tokens:
                if stop_seq in generated_text:
                    generated_text = generated_text.split(stop_seq)[0]
                    stop_token_used = stop_seq
                    break
            
            # Clean up special tokens
            generated_ids = self.tokenizer(generated_text)["input_ids"]
            generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            
            out.append(generated_text)
            
            # Score accuracy if ground truth is available (e.g., GSM8K)
            gt_answer = _extract_answer_number(elem["doc_answer"]) if elem["doc_answer"] else None
            pred_answer = _extract_answer_number(generated_text)
            is_correct = None
            if gt_answer is not None:
                is_correct = (pred_answer == gt_answer)
                num_scored += 1
                if is_correct:
                    num_correct += 1

            # Per-example W&B logging (rank 0 only)
            if self.use_wandb and self._rank == 0:
                log_dict = {
                    "generation/iteration": self.iteration,
                    "generation/example_idx": elem_idx,
                    "generation/prompt_length": prompt.shape[1],
                    "generation/generated_length": len(generated_ids),
                    "generation/total_length": prompt.shape[1] + len(generated_ids),
                    "generation/max_gen_length": gen_length,
                    "generation/stop_token_used": stop_token_used is not None,
                    "generation/generated_text": wandb.Html(
                        f"<p><strong>Prompt:</strong> {self.tokenizer.decode(prompt[0])}</p>"
                        f"<p><strong>Generated:</strong> {generated_text}</p>"
                    ),
                }
                if is_correct is not None:
                    log_dict.update({
                        "generation/correct": int(is_correct),
                        "generation/pred_answer": pred_answer or "",
                        "generation/gt_answer": gt_answer,
                        "generation/running_accuracy": num_correct / num_scored,
                    })
                wandb.log(log_dict)

                self.iteration += 1
        
        if self.accelerator is not None:
            self.accelerator.wait_for_everyone()
        
        # Log summary statistics (rank 0 only)
        if self.use_wandb and self._rank == 0 and len(out) > 0:
            avg_length = sum(len(self.tokenizer(text)["input_ids"]) for text in out) / len(out)

            summary = {
                "generation_summary/num_examples": len(out),
                "generation_summary/avg_generated_length": avg_length,
            }
            if num_scored > 0:
                summary["generation_summary/final_accuracy"] = num_correct / num_scored
                summary["generation_summary/num_correct"] = num_correct
                summary["generation_summary/num_scored"] = num_scored
            wandb.log(summary)
        
        return out


if __name__ == "__main__":
    cli_evaluate()