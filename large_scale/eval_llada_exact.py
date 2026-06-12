'''
This file is inspired by the code from https://github.com/ML-GSAI/SMDM

accelerate launch eval_llada_exact.py \
    --tasks gpqa_main_n_shot \
    --num_fewshot 5 \
    --model llada_exact \
    --batch_size 8 \
    --limit 8 \
    --model_args model_path='GSAI-ML/LLaDA-8B-Base',cfg=0.0,is_check_greedy=False,strategy='greedy'
'''
import warnings
warnings.filterwarnings('ignore', category=Warning, module='dill')
import math
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

from exact_likelihood.compute import compute_deterministic_loglikelihood
from exact_likelihood.strategies import (
    BlockGreedyConfidenceStrategy,
    BlockLeftToRightStrategy,
    BlockProbabilityMarginStrategy,
    ConfidenceThresholdStrategy,
    GreedyConfidenceStrategy,
    ProbabilityMarginStrategy,
)


def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@register_model("llada_exact")
class LLaDAEvalHarness(LM):
    def __init__(
        self,
        model_path='',
        mask_token_id=126336,
        pad_token_id=126081,
        max_length=1028,
        batch_size=32,
        is_check_greedy=False,
        cfg=0.,
        steps=1024,
        gen_length=1024,
        block_length=32,
        strategy='greedy',
        device="cuda",
        **kwargs,
    ):
        '''
        Args:
            model_path: LLaDA-8B-Base model path.
            mask_token_id: The token id of [MASK] is 126336.
            pad_token_id: The token id of [PAD] is 126081. Used for batching multiple sequences of varying length into a fixed-length tensor.
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

        self.mask_token_id = mask_token_id
        self.pad_token_id = pad_token_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        self.batch_size = int(batch_size)
        self.sampling_eps = 0.
        self.max_length = max_length
        self.is_check_greedy = is_check_greedy

        self.cfg = cfg
        self.steps = steps
        self.gen_length = gen_length
        self.block_length = block_length
        
        if strategy == 'greedy':
            self.strategy = GreedyConfidenceStrategy(k=1)
        elif strategy == 'block_greedy':
            self.strategy = BlockGreedyConfidenceStrategy(block_size=self.block_length, k=1)
        elif strategy == 'block_left_to_right':
            self.strategy = BlockLeftToRightStrategy(block_size=self.block_length, k=1)
        elif strategy == 'probability_margin':
            self.strategy = ProbabilityMarginStrategy(k=1)
        elif strategy == 'block_probability_margin':
            self.strategy = BlockProbabilityMarginStrategy(block_size=self.block_length, k=1)
        elif strategy == 'confidence_threshold':
            self.strategy = ConfidenceThresholdStrategy(threshold=0.9)
        else:
            raise ValueError(f'Unknown position decoding strategy: {self.strategy}')
        
        wandb.init(project="exact-ll", tags=[strategy])
    
    @property
    def rank(self):
        return self._rank
    
    @property
    def world_size(self):
        return self._world_size

    @torch.no_grad()
    def get_logits(self, batch, prompt_index, attention_mask=None):
        if self.cfg > 0.:
            assert len(prompt_index) == batch.shape[1]
            prompt_index = prompt_index.unsqueeze(0).repeat(batch.shape[0], 1)
            un_batch = batch.clone()
            un_batch[prompt_index] = self.mask_token_id
            batch = torch.cat([batch, un_batch])
            
            if attention_mask is not None:
                attention_mask = torch.cat([attention_mask, attention_mask])

        logits = self.model(batch, attention_mask=attention_mask).logits

        if self.cfg > 0.:
            logits, un_logits = torch.chunk(logits, 2, dim=0)
            logits = un_logits + (self.cfg + 1) * (logits - un_logits)
        return logits[:, :batch.shape[1]]

    @torch.no_grad()
    def get_loglikelihood(self, tokens, prompt_lens, lengths, attention_mask):
        """Compute deterministic loglikelihood with specified position decoding strategy.
        
        Assumes batch size of 1.
        
        Args:
            tokens: [seq_len]
            prompt_index: [seq_len] bool tensor, True for prompt tokens
        """
        
        ll = compute_deterministic_loglikelihood(
            model=self.model, 
            x=tokens, 
            prompt_lens=prompt_lens, 
            lengths=lengths, 
            mask_token_id=self.mask_token_id, 
            strategy=self.strategy,
            attention_mask=attention_mask
        )
        
        for idx in range(len(ll)):
            nll_normalized = -ll[idx].item() / (lengths[idx] - prompt_lens[idx])
            wandb.log({("NLL_normalized"): nll_normalized, "Perplexity_normalized": math.exp(nll_normalized)})
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
    
    def _create_batch(self, elements, include_target=True):
        """Create a padded batch from list of elements.
        
        Args:
            elements: List of dicts with keys 'prefix', 'target' (optional)
            include_target: Whether to include target in the sequence
        
        Returns:
            tokens: [batch_size, max_seq_len] padded tensor
            prompt_lens: [batch_size] tensor
            lengths: [batch_size] tensor
        """
        batch_size = len(elements)
        
        # Build sequences and compute lengths
        sequences = []
        prompt_lens = []
        lengths = []
        
        for elem in elements:
            if include_target:
                prefix = torch.tensor(elem["prefix"]) if not isinstance(elem["prefix"], torch.Tensor) else elem["prefix"]
                target = torch.tensor(elem["target"]) if not isinstance(elem["target"], torch.Tensor) else elem["target"]
                seq = torch.cat([prefix, target])
                prompt_len = len(prefix)
            else:
                tokens = elem["tokens"]
                seq = torch.tensor(tokens) if not isinstance(tokens, torch.Tensor) else tokens
                prompt_len = 0
            
            sequences.append(seq)
            prompt_lens.append(prompt_len)
            lengths.append(len(seq))
        
        # Pad to max length in batch
        max_len = max(lengths)
        tokens = torch.full(
            (batch_size, max_len), 
            self.pad_token_id, 
            dtype=torch.long, 
            device=self.device
        )
        
        for i, seq in enumerate(sequences):
            tokens[i, :len(seq)] = seq.to(self.device)
        
        prompt_lens = torch.tensor(prompt_lens, dtype=torch.long, device=self.device)
        lengths = torch.tensor(lengths, dtype=torch.long, device=self.device)
        
        attention_mask = (tokens != self.pad_token_id).long()
        
        return tokens, prompt_lens, lengths, attention_mask

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
        print(f"Total number of examples: {len(ds)}")
        
        max_seq_length = max([len(x["prefix"]) + len(x["target"]) for x in ds])
        if max_seq_length > self.max_length:
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
        num_batches = math.ceil(len(ds) / self.batch_size)
        
        with torch.no_grad():
            for batch_idx in tqdm(range(num_batches), desc="Computing likelihood..."):
                
                # Calculate batch start and end indices
                batch_start = batch_idx * self.batch_size
                batch_end = min((batch_idx + 1) * self.batch_size, len(ds))
                batch_elements = [ds[i] for i in range(batch_start, batch_end)]
                
                # Create batch and compute loglikelihoods
                tokens, prompt_lens, lengths, attention_mask = self._create_batch(batch_elements, include_target=True) 
                ll_batch = self.get_loglikelihood(tokens, prompt_lens, lengths, attention_mask=attention_mask)
                
                # Compute greedy predictions if needed
                for i, elem in enumerate(batch_elements):
                    ll = ll_batch[i].item()
                    is_greedy = self.suffix_greedy_prediction(elem["prefix"], elem["target"])
                    out.append((ll, 1.0 if is_greedy else 0.0))
        
        torch.cuda.empty_cache()
        return out
    
    def loglikelihood_rolling(self, requests):
        """Compute rolling log-likelihood by chunking documents and batching chunks."""
        
        # Tokenize all documents upfront
        print("Tokenizing documents...")
        all_tokens = [
            self.tokenizer(req.args[0], )["input_ids"] 
            for req in requests
        ]
        
        # ds = Dataset.from_dict({
        #     "text": [req.args[0] for req in requests]
        # })
        # ds = ds.map(
        #     lambda x: {"tokens": self.tokenizer(x["text"])["input_ids"]},
        #     num_proc=min(4, len(requests)),
        #     desc="Tokenizing documents"
        # )
        print("Chunking documents...")
        
        # Create all chunks across all documents
        all_chunks = []
        doc_chunk_counts = []
        
        # for tokens in ds["tokens"]:
        for tokens in all_tokens:
            num_chunks = math.ceil(len(tokens) / self.max_length)
            doc_chunk_counts.append(num_chunks)
            
            for start_idx in range(0, len(tokens), self.max_length):
                chunk_tokens = tokens[start_idx:start_idx + self.max_length]
                all_chunks.append({"tokens": chunk_tokens})
        
        print(f"Total documents: {len(requests)}")
        print(f"Total chunks: {len(all_chunks)}")
        print(f"Average chunks per document: {len(all_chunks) / len(requests):.2f}")
        
        # Batch process all chunks
        chunk_lls = []
        num_batches = math.ceil(len(all_chunks) / self.batch_size)
        
        with torch.no_grad():
            for batch_idx in tqdm(range(num_batches), desc="Computing rolling likelihood..."):
                batch_start = batch_idx * self.batch_size
                batch_end = min((batch_idx + 1) * self.batch_size, len(all_chunks))
                batch_elements = all_chunks[batch_start:batch_end]
                
                tokens, prompt_lens, lengths, attention_mask = self._create_batch(
                    batch_elements, include_target=False
                )
                ll_batch = self.get_loglikelihood(tokens, prompt_lens, lengths, attention_mask)
                chunk_lls.extend([ll.item() for ll in ll_batch])
        
        # Aggregate chunks back to documents
        out = []
        chunk_idx = 0
        for num_chunks in doc_chunk_counts:
            doc_ll = sum(chunk_lls[chunk_idx:chunk_idx + num_chunks])
            out.append(doc_ll)
            chunk_idx += num_chunks
        
        torch.cuda.empty_cache()
        return out

    def generate_until(self, requests: list[Instance]):
        raise NotImplementedError


if __name__ == "__main__":
    set_seed(1234)
    cli_evaluate()
