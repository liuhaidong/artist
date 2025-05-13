# models/policy_model.py
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List, Optional, Tuple, Union

class PolicyModel:
    def __init__(self, config):
        self.config = config
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
    def generate(self, 
                prompt: str, 
                max_new_tokens: int = None, 
                temperature: float = None,
                do_sample: bool = True) -> str:
        """
        Generate text from the policy model given a prompt.
        
        Args:
            prompt: The input prompt
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature (higher = more random)
            do_sample: Whether to use sampling (vs greedy decoding)
            
        Returns:
            Generated text
        """
        if max_new_tokens is None:
            max_new_tokens = self.config.max_length
        if temperature is None:
            temperature = self.config.temperature
            
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample and temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
                top_p=0.95,  # Nucleus sampling
                repetition_penalty=1.1  # Discourage repetition
            )
            
        response = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:], 
            skip_special_tokens=True
        )
        return response
    
    def get_logprobs(self, 
                    input_ids: torch.Tensor, 
                    attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Get log probabilities from the model for given input.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            
        Returns:
            Log probabilities tensor of shape [batch, seq_len, vocab_size]
        """
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
        logits = outputs.logits
        log_probs = F.log_softmax(logits, dim=-1)
        
        return log_probs
    
    def save(self, path: str):
        """Save the policy model."""
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
