'''
Llama3 evaluation harness for lm-eval

accelerate launch eval_llama3.py \
    --tasks hellaswag \
    --num_fewshot 10 \
    --model llama3 \
    --batch_size 8 \
    --limit 100 \
    --model_args model_path='meta-llama/Meta-Llama-3-8B',dtype='bfloat16'

# For instruct model:
# --model_args model_path='meta-llama/Meta-Llama-3-8B-Instruct',dtype='bfloat16'
'''
import warnings
warnings.filterwarnings('ignore', category=Warning, module='dill')
import math
import accelerate
import torch
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
from transformers import AutoTokenizer, AutoModelForCausalLM


def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@register_model("llama3")
class Llama3EvalHarness(LM):
    def __init__(
        self,
        model_path='meta-llama/Meta-Llama-3-8B',
        max_length=2048,
        batch_size=8,
        dtype='bfloat16',
        device="cuda",
        **kwargs,
    ):
        '''
        Args:
            model_path: HuggingFace model path (supports both base and instruct)
            max_length: Maximum sequence length
            batch_size: Batch size for evaluation
            dtype: Model dtype ('bfloat16', 'float16', or 'float32')
            device: Device to run on
        '''
        super().__init__()

        # Setup accelerate for distributed evaluation
        accelerator = accelerate.Accelerator()
        if accelerator.num_processes > 1:
            self.accelerator = accelerator
        else:
            self.accelerator = None
        
        # Determine dtype
        dtype_map = {
            'bfloat16': torch.bfloat16,
            'float16': torch.float16,
            'float32': torch.float32,
        }
        dtype = dtype_map.get(dtype, torch.bfloat16)
        
        # Model loading kwargs
        model_kwargs = {'dtype': dtype}
        if self.accelerator is not None:
            model_kwargs['device_map'] = {'': self.accelerator.device}
        else:
            model_kwargs['device_map'] = device
        
        # Load model and tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            **model_kwargs
        )
        self.model.eval()
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Llama3 doesn't have pad token by default, set to eos
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # Setup device and distributed settings
        self.device = torch.device(device)
        if self.accelerator is not None:
            self.model = self.accelerator.prepare(self.model)
            self.device = self.accelerator.device
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self._rank = 0
            self._world_size = 1
        
        self.max_length = int(max_length)
        self.batch_size = int(batch_size)
        
        # Initialize wandb (only on main process)
        if self._rank == 0:
            wandb.init(project="llama3-eval", tags=["autoregressive"])
    
    @property
    def rank(self):
        return self._rank
    
    @property
    def world_size(self):
        return self._world_size
    
    @property
    def eot_token_id(self):
        """End of text token for compatibility with lm-eval"""
        return self.tokenizer.eos_token_id
    
    @property
    def max_gen_toks(self):
        """Maximum generation tokens for compatibility"""
        return self.max_length

    def _encode_pair(self, context, continuation):
        """
        Tokenize context and continuation separately, handling whitespace correctly.
        
        Design choice: This matches your existing implementation and handles the
        common case where context has trailing whitespace that should be part of
        the continuation for correct tokenization.
        """
        n_spaces = len(context) - len(context.rstrip())
        if n_spaces > 0:
            continuation = context[-n_spaces:] + continuation
            context = context[:-n_spaces]

        whole_enc = self.tokenizer(context + continuation, add_special_tokens=True)["input_ids"]
        context_enc = self.tokenizer(context, add_special_tokens=True)["input_ids"]

        context_enc_len = len(context_enc)
        continuation_enc = whole_enc[context_enc_len:]

        return context_enc, continuation_enc
    
    def _create_batch(self, elements, include_target=True):
        """
        Create a left-padded batch for causal language modeling.
        
        Design choice: Left-padding is critical for autoregressive models because:
        1. Causal attention mask assumes padding is on the left
        2. Ensures the final token position aligns across batch
        3. Makes generation cleaner (last position is always the continuation end)
        
        Args:
            elements: List of dicts with 'prefix' and 'target' keys
            include_target: If True, concatenate prefix+target; else use 'tokens' key
        
        Returns:
            input_ids: [batch_size, max_seq_len] left-padded tensor
            attention_mask: [batch_size, max_seq_len] mask tensor
            prompt_lens: [batch_size] tensor of prompt lengths
            lengths: [batch_size] tensor of total sequence lengths
        """
        batch_size = len(elements)
        
        sequences = []
        prompt_lens = []
        
        for elem in elements:
            if include_target:
                # Concatenate prefix and target
                prefix = elem["prefix"] if isinstance(elem["prefix"], list) else elem["prefix"].tolist()
                target = elem["target"] if isinstance(elem["target"], list) else elem["target"].tolist()
                seq = prefix + target
                prompt_len = len(prefix)
            else:
                # Use full sequence (for rolling likelihood)
                tokens = elem["tokens"]
                seq = tokens if isinstance(tokens, list) else tokens.tolist()
                prompt_len = 0
            
            sequences.append(seq)
            prompt_lens.append(prompt_len)
        
        # Get max length
        lengths = [len(seq) for seq in sequences]
        max_len = max(lengths)
        
        # Left-pad sequences (critical for causal LM)
        input_ids = torch.full(
            (batch_size, max_len),
            self.tokenizer.pad_token_id,
            dtype=torch.long,
            device=self.device
        )
        
        for i, seq in enumerate(sequences):
            # Place sequence at the right (left-padding)
            seq_len = len(seq)
            input_ids[i, -seq_len:] = torch.tensor(seq, dtype=torch.long, device=self.device)
        
        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
        
        # Adjust prompt_lens for left-padding offset
        prompt_lens_adjusted = []
        for i, (prompt_len, seq_len) in enumerate(zip(prompt_lens, lengths)):
            pad_len = max_len - seq_len
            # Prompt length in the padded sequence
            prompt_lens_adjusted.append(pad_len + prompt_len)
        
        prompt_lens = torch.tensor(prompt_lens_adjusted, dtype=torch.long, device=self.device)
        lengths = torch.tensor(lengths, dtype=torch.long, device=self.device)
        
        return input_ids, attention_mask, prompt_lens, lengths

    @torch.no_grad()
    def get_loglikelihood(self, input_ids, attention_mask, prompt_lens, lengths):
        """
        Compute log-likelihood for autoregressive model.
        
        Design choice: For causal LM, logits at position i predict token at position i+1.
        We only compute loss on continuation tokens (after prompt).
        
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            prompt_lens: [batch_size] - where prompts end in the padded sequence
            lengths: [batch_size] - actual sequence lengths (excluding padding)
        
        Returns:
            log_likelihoods: [batch_size] tensor of log-likelihoods
        """
        # Forward pass
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # [batch_size, seq_len, vocab_size]
        
        # Shift for next-token prediction
        # logits[:, :-1] predicts input_ids[:, 1:]
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        
        batch_size, seq_len_minus_1, vocab_size = shift_logits.shape
        
        # Compute per-token log probabilities
        log_probs = F.log_softmax(shift_logits, dim=-1)
        
        # Gather log probs of actual tokens
        # Shape: [batch_size, seq_len - 1]
        token_log_probs = torch.gather(
            log_probs,
            dim=-1,
            index=shift_labels.unsqueeze(-1)
        ).squeeze(-1)
        
        # Create mask for continuation tokens only
        # We want to sum log probs only for tokens after the prompt
        continuation_mask = torch.zeros(batch_size, seq_len_minus_1, dtype=torch.bool, device=self.device)
        
        for i in range(batch_size):
            # Prompt ends at prompt_lens[i], so continuation starts at prompt_lens[i]
            # In shifted coordinates (seq_len - 1), continuation starts at prompt_lens[i]
            # and ends at the actual sequence end
            start_idx = prompt_lens[i]
            end_idx = input_ids.shape[1] - 1  # Last position in shifted sequence
            
            # Only include positions that are part of the actual sequence (not padding)
            actual_end = start_idx + (lengths[i] - prompt_lens[i] + (input_ids.shape[1] - lengths[i] - (input_ids.shape[1] - prompt_lens[i] - lengths[i])))
            # Simpler: the continuation in original coords is from prompt_lens[i] to end
            # In shifted coords, we need positions from prompt_lens[i] onwards
            continuation_mask[i, start_idx:] = True
            
            # But also mask out padding (which is at the beginning due to left-padding)
            pad_length = input_ids.shape[1] - lengths[i]
            if pad_length > 0:
                continuation_mask[i, :pad_length] = False
        
        # Sum log probs over continuation tokens
        masked_log_probs = token_log_probs * continuation_mask.float()
        log_likelihoods = masked_log_probs.sum(dim=-1)
        
        # Log normalized perplexity (per-token NLL)
        if self._rank == 0:
            for i in range(batch_size):
                num_continuation_tokens = continuation_mask[i].sum().item()
                if num_continuation_tokens > 0:
                    nll_normalized = -log_likelihoods[i].item() / num_continuation_tokens
                    wandb.log({
                        "NLL_normalized": nll_normalized,
                        "Perplexity_normalized": math.exp(nll_normalized)
                    })
        
        return log_likelihoods

    def loglikelihood(self, requests):
        """
        Compute log-likelihood for multiple context-continuation pairs.
        
        Design choice: Batch processing for efficiency, with clean tokenization
        using HF datasets (no multiprocessing to avoid conflicts with accelerate).
        """
        # Tokenize all requests
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
        ds = ds.map(_tokenize)
        print(f"Total number of examples: {len(ds)}")
        
        # Check for sequences exceeding max_length
        max_seq_length = max([len(x["prefix"]) + len(x["target"]) for x in ds])
        if max_seq_length > self.max_length:
            print(f"Warning: some sequences exceed max_length {self.max_length}, truncating.")
            ds = ds.map(lambda x: {
                "prefix": x["prefix"][:self.max_length],
                "target": x["target"][:self.max_length - len(x["prefix"])]
            })
        
        # Batch processing
        out = []
        num_batches = math.ceil(len(ds) / self.batch_size)
        
        with torch.no_grad():
            for batch_idx in tqdm(range(num_batches), desc="Computing likelihood..."):
                batch_start = batch_idx * self.batch_size
                batch_end = min((batch_idx + 1) * self.batch_size, len(ds))
                batch_elements = [ds[i] for i in range(batch_start, batch_end)]
                
                # Create batch
                input_ids, attention_mask, prompt_lens, lengths = self._create_batch(
                    batch_elements, include_target=True
                )
                
                # Compute log-likelihoods
                ll_batch = self.get_loglikelihood(input_ids, attention_mask, prompt_lens, lengths)
                
                # Store results (greedy is always 0 for standard LM)
                for ll in ll_batch:
                    out.append((ll.item(), 0.0))
        
        torch.cuda.empty_cache()
        return out
    
    def loglikelihood_rolling(self, requests):
        """
        Compute rolling log-likelihood by chunking documents.
        
        Design choice: 
        1. Batch tokenization (fast, no multiprocessing issues)
        2. Chunk documents into max_length segments
        3. Batch process chunks on GPU
        4. Aggregate chunk log-likelihoods back to documents
        """
        # Batch tokenize all documents
        print("Tokenizing documents...")
        texts = [req.args[0] for req in requests]
        all_tokens = self.tokenizer(
            texts,
            add_special_tokens=True,
            return_tensors=None,
            padding=False,
            truncation=False
        )["input_ids"]
        
        # Create chunks
        print("Chunking documents...")
        all_chunks = []
        doc_chunk_counts = []
        
        for tokens in all_tokens:
            num_chunks = math.ceil(len(tokens) / self.max_length)
            doc_chunk_counts.append(num_chunks)
            
            for start_idx in range(0, len(tokens), self.max_length):
                chunk_tokens = tokens[start_idx:start_idx + self.max_length]
                all_chunks.append({"tokens": chunk_tokens})
        
        print(f"Total documents: {len(requests)}")
        print(f"Total chunks: {len(all_chunks)}")
        print(f"Average chunks per document: {len(all_chunks) / len(requests):.2f}")
        
        # Batch process chunks
        chunk_lls = []
        num_batches = math.ceil(len(all_chunks) / self.batch_size)
        
        with torch.no_grad():
            for batch_idx in tqdm(range(num_batches), desc="Computing rolling likelihood..."):
                batch_start = batch_idx * self.batch_size
                batch_end = min((batch_idx + 1) * self.batch_size, len(all_chunks))
                batch_elements = all_chunks[batch_start:batch_end]
                
                # Create batch (prompt_len = 0 for rolling)
                input_ids, attention_mask, prompt_lens, lengths = self._create_batch(
                    batch_elements, include_target=False
                )
                
                # Compute log-likelihoods
                ll_batch = self.get_loglikelihood(input_ids, attention_mask, prompt_lens, lengths)
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
        """
        Not implemented for this evaluation (only likelihood-based metrics).
        """
        raise NotImplementedError("Generation not implemented for this harness")


if __name__ == "__main__":
    set_seed(1234)
    cli_evaluate()