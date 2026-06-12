'''
This file is inspired by the code from https://github.com/ML-GSAI/SMDM

accelerate launch eval_llada_elbo.py \
    --tasks gpqa_main_n_shot \
    --num_fewshot 5 \
    --model llada_elbo \
    --batch_size 8 \
    --limit 8 \
    --model_args model_path='GSAI-ML/LLaDA-8B-Base',cfg=0.0,is_check_greedy=False,mc_num=8
'''
import warnings
warnings.filterwarnings('ignore', category=Warning, module='dill')
import math
import sys
import accelerate
import torch
import re
from pathlib import Path
import random
import numpy as np
import wandb
import torch.nn.functional as F
from datasets import Dataset
from lm_eval.__main__ import cli_evaluate
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

from generate import generate


def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@register_model("llada_elbo")
class LLaDAEvalHarness(LM):
    def __init__(
        self,
        model_path='',
        mask_token_id=126336,
        max_length=1028,
        batch_size=32,
        mc_num=128,
        is_check_greedy=False,
        cfg=0.,
        steps=1024,
        gen_length=1024,
        block_length=32,
        k=1,
        remasking='random',
        device="cuda",
        use_wandb=True,
        wandb_project="llada-exact-likelihood",
        wandb_entity=None,
        wandb_run_name=None,
        wandb_tags=None,
        **kwargs,
    ):
        '''
        Args:
            model_path: LLaDA-8B-Base model path.
            mask_token_id: The token id of [MASK] is 126336.
            max_length: the max sequence length.
            batch_size: mini batch size.
            mc_num: Monte Carlo estimation iterations
            is_check_greedy: For certain metrics like LAMBADA, the evaluation requires the model to verify whether the answer 
                             is generated through greedy sampling conditioned on the prompt (note that this differs from conditional
                             generation). We implement this verification through the suffix_greedy_prediction() function, which 
                             returns a True/False judgment used for accuracy calculation. 
                             When is_check_greedy is set to True, the lm-evaluation-harness library automatically invokes this function. 
                             However, since none of the metrics in the LLaDA paper (https://arxiv.org/abs/2502.09992) require this functionality, 
                             we recommend setting is_check_greedy to False. This configuration causes suffix_greedy_prediction() to return False 
                             by default, significantly accelerating the evaluation process.
            cfg_scale: Unsupervised classifier-free guidance scale.
        '''
        super().__init__()

        accelerator = accelerate.Accelerator()
        if accelerator.num_processes > 1:
            self.accelerator = accelerator
        else:
            self.accelerator = None
        
        model_kwargs = {}
        if self.accelerator is not None:
            model_kwargs.update({'device_map': {'': f'{self.accelerator.device}'}})

        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True, dtype=torch.bfloat16, **model_kwargs)
        self.model.eval()

        self.device = torch.device(device)
        if self.accelerator is not None:
            self.model = self.accelerator.prepare(self.model)
            self.device = torch.device(f'{self.accelerator.device}')
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self.model = self.model.to(device)
            self._rank = 0
            self._world_size = 1

        self.mask_token_id = mask_token_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        self.mc_num = mc_num
        self.batch_size = int(batch_size)
        assert mc_num % self.batch_size == 0 , f"mc_num ({mc_num}) must be divisible by batch_size ({self.batch_size})"
        self.sampling_eps = 0.
        self.max_length = max_length
        self.is_check_greedy = is_check_greedy

        self.cfg = cfg
        self.gen_length = gen_length
        self.block_length = int(block_length)
        self.remasking = remasking

        # Auto-set steps so that steps_per_block = block_length / k.
        # This avoids wasted forward passes when gen_length < steps.
        self.k = int(k)
        self.steps = gen_length // self.k

        # Cast use_wandb from string if needed (lm-eval passes strings)
        if isinstance(use_wandb, str):
            use_wandb = use_wandb.lower() not in ('false', '0', 'no')

        # Initialize Weights & Biases (only on rank 0)
        self.use_wandb = use_wandb
        if self.use_wandb and self._rank == 0:
            # Parse wandb tags
            # Extract task name from CLI args
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
                wandb_tags = ["elbo", f"mc_num_{mc_num}", f"k_{self.k}", task_type]
                if limit is not None:
                    wandb_tags.append(f"limit_{limit}")
            elif isinstance(wandb_tags, str):
                wandb_tags = [t.strip() for t in wandb_tags.split(',')]

            # Auto-generate run name if not provided
            if wandb_run_name is None:
                limit_suffix = f"_lim{limit}" if limit is not None else ""
                wandb_run_name = f"elbo_k{self.k}_mc{mc_num}_batch{self.batch_size}_{task_name}{limit_suffix}"

            wandb.init(
                project=wandb_project,
                entity=wandb_entity,
                name=wandb_run_name,
                tags=wandb_tags,
                config={
                    "model_path": model_path,
                    "strategy": "elbo",
                    "mc_num": mc_num,
                    "k": self.k,
                    "batch_size": self.batch_size,
                    "gen_length": gen_length,
                    "steps": self.steps,
                    "block_length": self.block_length,
                    "max_length": max_length,
                    "mask_token_id": mask_token_id,
                    "num_gpus": self._world_size,
                    "limit": limit,
                }
            )

            print(f"[Rank {self._rank}] Initialized W&B: {wandb.run.name}")

        # Iteration counter for logging
        self.iteration = 0
    
    @property
    def rank(self):
        return self._rank
    
    @property
    def world_size(self):
        return self._world_size

    def __del__(self):
        """Cleanup: finish wandb run if active."""
        if getattr(self, 'use_wandb', False) and getattr(self, '_rank', 1) == 0 and wandb.run is not None:
            wandb.finish()

    def _forward_process(self, batch, prompt_index):
        b, l = batch.shape

        target_len = (l - prompt_index.sum()).item()
        k = torch.randint(1, target_len + 1, (), device=batch.device)

        x = torch.round(torch.linspace(float(k), k + (b - 1) * (target_len / b), steps=b, device=batch.device)).long()
        x = ((x - 1) % target_len) + 1
        assert x.min() >= 1 and x.max() <= target_len

        indices = torch.arange(target_len, device=batch.device).repeat(b, 1)
        is_mask = indices < x.unsqueeze(1)

        for i in range(b):
            is_mask[i] = is_mask[i][torch.randperm(target_len)]

        is_mask = torch.cat((torch.zeros(b, prompt_index.sum(), dtype=torch.bool, device=batch.device), is_mask), dim=1)

        noisy_batch = torch.where(is_mask, self.mask_token_id, batch)

        return noisy_batch, (x / target_len).unsqueeze(1).repeat(1, l)

    @torch.no_grad()
    def get_logits(self, batch, prompt_index):
        if self.cfg > 0.:
            assert len(prompt_index) == batch.shape[1]
            prompt_index = prompt_index.unsqueeze(0).repeat(batch.shape[0], 1)
            un_batch = batch.clone()
            un_batch[prompt_index] = self.mask_token_id
            batch = torch.cat([batch, un_batch])

        logits = self.model(batch).logits

        if self.cfg > 0.:
            logits, un_logits = torch.chunk(logits, 2, dim=0)
            logits = un_logits + (self.cfg + 1) * (logits - un_logits)
        return logits[:, :batch.shape[1]]

    @torch.no_grad()
    def get_loglikelihood(self, tokens, prompt_index):
        """Compute log-likelihood using Monte Carlo sampling.
        
        Note: computes micro-averaged log-likelihood instead of macro-averaged.
        This is incorrect but consistent with prior work (LLaDA paper).
        """
        seq = tokens[None, :].repeat((self.batch_size, 1)).to(self.device)
        prompt_index = prompt_index.to(self.device)
        
        loss_acc = []
        for _ in range(self.mc_num // self.batch_size):
            perturbed_seq, p_mask = self._forward_process(seq, prompt_index)
            mask_indices = perturbed_seq == self.mask_token_id
            logits = self.get_logits(perturbed_seq, prompt_index)
            
            loss = F.cross_entropy(
                logits[mask_indices],
                seq[mask_indices], 
                reduction='none'
            ) / p_mask[mask_indices]
            loss = loss.sum() / self.batch_size
            loss_acc.append(loss.item())
        
        ll = -sum(loss_acc) / len(loss_acc)
        return ll

    @torch.no_grad()
    def suffix_greedy_prediction(self, prefix, target):
        if not self.is_check_greedy:
            return False

        seq = torch.full((1, len(prefix) + len(target)), self.mask_token_id, device=self.device)
        prompt_index = torch.arange(seq.shape[1], device=self.device) < len(prefix)
        prefix, target = prefix.to(self.device), target.to(self.device)
        seq[0, :len(prefix)] = prefix

        for i in range(len(target)):
            mask_index = (seq == self.mask_token_id)
            logits = self.get_logits(seq, prompt_index)[mask_index]
            x0 = torch.argmax(logits, dim=-1)

            p = torch.softmax(logits.to(torch.float32), dim=-1)
            confidence = torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)).squeeze(dim=-1)
            _, index = torch.sort(confidence, descending=True)
            x0[index[1:]] = self.mask_token_id
            seq[mask_index] = x0.clone()
        correct = target == seq[0, len(prefix):]
        correct = torch.all(correct)
        return correct

    def _encode_pair(self, context, continuation):
        n_spaces = len(context) - len(context.rstrip())
        if n_spaces > 0:
            continuation = context[-n_spaces:] + continuation
            context = context[:-n_spaces]

        whole_enc = self.tokenizer(context + continuation)["input_ids"]
        context_enc = self.tokenizer(context)["input_ids"]

        context_enc_len = len(context_enc)
        continuation_enc = whole_enc[context_enc_len:]

        return context_enc, continuation_enc

    def loglikelihood(self, requests):
        def _tokenize(e):
            prefix, target = self._encode_pair(e["prefix"], e["target"])
            return {
                "prefix_text": e["prefix"],
                "target_text": e["target"],
                "prefix": prefix,
                "target": target,
            }

        ds = [{"prefix": req.args[0], "target": req.args[1]} for req in requests]
        ds = Dataset.from_list(ds)
        ds = ds.map(function=_tokenize)
        ds = ds.with_format("torch")
        
        max_seq_length = max([len(x["prefix"]) + len(x["target"]) for x in ds])
        if max_seq_length > self.max_length:
            # print(f"Warning: some sequences exceed max_length {self.max_length}, truncating.")
            # ds = ds.map(lambda x: {
            #     "prefix": x["prefix"][:self.max_length],
            #     "target": x["target"][:self.max_length - len(x["prefix"])]
            # })
            
            # Truncate to fit, prioritizing target
            ds = ds.map(lambda x: {
                "prefix": x["prefix"][max(0, len(x["prefix"]) - (self.max_length - len(x["target"]))):],
                "target": x["target"][:self.max_length]  # Ensure target also fits
            })

            # Warn about significant truncations
            truncated = [(i, len(x["prefix"]), len(x["target"])) 
                        for i, x in enumerate(ds) 
                        if len(x["prefix"]) + len(x["target"]) > self.max_length]
            if truncated:
                print(f"Warning: {len(truncated)} examples truncated. Prefix truncated from left to fit.")

        out = []
        with torch.no_grad():
            for example_idx, elem in enumerate(tqdm(ds, desc="Computing likelihood...")):
                # Concatenate prefix and target
                tokens = torch.cat([elem["prefix"], elem["target"]])
                prompt_len = len(elem["prefix"])
                answer_len = len(elem["target"])

                # Mark which tokens are prompt (all prefix tokens)
                prompt_index = torch.cat([
                    torch.ones(prompt_len, dtype=torch.bool),
                    torch.zeros(answer_len, dtype=torch.bool)
                ])

                ll = self.get_loglikelihood(tokens, prompt_index)
                is_greedy = self.suffix_greedy_prediction(elem["prefix"], elem["target"])
                out.append((ll, 1.0 if is_greedy else 0.0))

                # Per-example W&B logging (rank 0 only)
                if self.use_wandb and self._rank == 0:
                    nll_per_token = -ll / answer_len
                    ppl = math.exp(nll_per_token)

                    log_data = {
                        "iteration": self.iteration,
                        "example_idx": example_idx,
                        "log_likelihood": ll,
                        "perplexity": ppl,
                        "answer_length": answer_len,
                        "prompt_length": prompt_len,
                        "total_length": len(tokens),
                        "nll_per_token": nll_per_token,
                    }

                    if "prefix_text" in elem:
                        log_data["question"] = elem["prefix_text"]
                    if "target_text" in elem:
                        log_data["answer"] = elem["target_text"]

                    wandb.log(log_data)
                    self.iteration += 1

        # Log summary statistics (rank 0 only)
        if self.use_wandb and self._rank == 0 and len(out) > 0:
            lls = [x[0] for x in out]
            wandb.log({
                "summary/mean_ll": sum(lls) / len(lls),
                "summary/num_examples": len(lls),
            })

        torch.cuda.empty_cache()
        return out

    def loglikelihood_rolling(self, requests):
        """Compute rolling log-likelihood by chunking documents into max_length segments."""

        if self._rank == 0:
            print(f"Total documents: {len(requests)}")

        # Tokenize all documents upfront (CPU work done first)
        all_tokens = [
            self.tokenizer(req.args[0])["input_ids"]
            for req in tqdm(requests, desc="Tokenizing documents...", disable=self._rank != 0)
        ]

        # Process chunks (GPU work)
        out = []
        total_chunks = 0

        with torch.no_grad():
            for doc_idx, tokens in enumerate(tqdm(all_tokens, desc="Computing rolling likelihood...", disable=self._rank != 0)):
                doc_ll = 0.0
                num_chunks = 0

                for start_idx in range(0, len(tokens), self.max_length):
                    chunk_tokens = torch.tensor(
                        tokens[start_idx:start_idx + self.max_length],
                        dtype=torch.long
                    )
                    prompt_index = torch.zeros(len(chunk_tokens), dtype=torch.bool)

                    chunk_ll = self.get_loglikelihood(chunk_tokens, prompt_index)
                    doc_ll += chunk_ll
                    num_chunks += 1

                    # Per-chunk W&B logging (rank 0 only)
                    if self.use_wandb and self._rank == 0:
                        chunk_ppl = math.exp(-chunk_ll / len(chunk_tokens))

                        wandb.log({
                            "rolling/iteration": self.iteration,
                            "rolling/doc_idx": doc_idx,
                            "rolling/chunk_idx": num_chunks - 1,
                            "rolling/chunk_ll": chunk_ll,
                            "rolling/chunk_ppl": chunk_ppl,
                            "rolling/chunk_length": len(chunk_tokens),
                        })

                        self.iteration += 1

                out.append(doc_ll)
                total_chunks += num_chunks

                # Per-document W&B logging (rank 0 only)
                if self.use_wandb and self._rank == 0:
                    doc_ppl = math.exp(-doc_ll / len(tokens))

                    wandb.log({
                        "rolling/doc_idx": doc_idx,
                        "rolling/doc_ll": doc_ll,
                        "rolling/doc_ppl": doc_ppl,
                        "rolling/doc_length": len(tokens),
                        "rolling/num_chunks": num_chunks,
                    })

        if self._rank == 0:
            print(f"Total chunks processed: {total_chunks}")
            print(f"Average chunks per document: {total_chunks / len(requests):.2f}")

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

        torch.cuda.empty_cache()
        return out

    def generate_until(self, requests: list[Instance]):
        ANSWER_PATTERN = re.compile(r"####\s*(\-?[0-9\.\,]+)")

        def _extract_answer_number(text):
            """Extract the number after #### from GSM8K-style answers."""
            match = ANSWER_PATTERN.search(text)
            if match:
                return match.group(1).replace(",", "")
            return None

        def _tokenize(e):
            return {
                "question": self.tokenizer(e["question"])["input_ids"],
                "question_text": e["question"],
                "until": e["until"],
                "doc_answer": e["doc_answer"],
            }

        ds = [{"question": req.args[0], "until": req.args[1]['until'],
               "doc_answer": req.doc.get("answer", "") if hasattr(req, "doc") and req.doc else ""}
              for req in requests]
        ds = Dataset.from_list(ds)
        ds = ds.map(_tokenize)
        ds = ds.with_format("torch")

        out = []
        num_correct = 0
        num_scored = 0
        desc = f"Generating (Rank {self._rank})" if self._world_size > 1 else "Generating"

        for elem_idx, elem in enumerate(tqdm(ds, desc=desc, disable=self._rank != 0)):
            prompt = elem["question"].unsqueeze(0).to(self.device)
            stop_tokens = elem["until"]

            generated_answer = generate(self.model, prompt, steps=self.steps, gen_length=self.gen_length, block_length=self.block_length,
                                        temperature=0, cfg_scale=self.cfg, remasking=self.remasking, mask_id=self.mask_token_id)

            generated_answer = self.tokenizer.decode(generated_answer[0][prompt.shape[1]:], skip_special_tokens=False)

            stop_token_used = None
            for stop_seq in stop_tokens:
                if stop_seq in generated_answer:
                    generated_answer = generated_answer.split(stop_seq)[0]
                    stop_token_used = stop_seq
                    break

            # remove special tokens
            generated_answer_ids = self.tokenizer(generated_answer)["input_ids"]
            generated_answer = self.tokenizer.decode(generated_answer_ids, skip_special_tokens=True)
            out.append(generated_answer)

            # Score accuracy if ground truth is available (e.g., GSM8K)
            gt_answer = _extract_answer_number(elem["doc_answer"]) if elem["doc_answer"] else None
            pred_answer = _extract_answer_number(generated_answer)
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
                    "generation/generated_length": len(generated_answer_ids),
                    "generation/total_length": prompt.shape[1] + len(generated_answer_ids),
                    "generation/max_gen_length": self.gen_length,
                    "generation/stop_token_used": stop_token_used is not None,
                    "generation/generated_text": wandb.Html(
                        f"<p><strong>Prompt:</strong> {self.tokenizer.decode(prompt[0])}</p>"
                        f"<p><strong>Generated:</strong> {generated_answer}</p>"
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
    set_seed(1234)
    cli_evaluate()
    