# models/policy_model.py
import torch
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
        
    def generate(self, prompt: str, max_new_tokens: int = None, temperature: float = None) -> str:
        """Generate text from the policy model given a prompt."""
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
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id
            )
            
        response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        return response
    
    def compute_logprobs(self, input_ids, attention_mask, labels):
        """Compute log probabilities for GRPO optimization."""
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
        logits = outputs.logits
        log_probs = torch.log_softmax(logits, dim=-1)
        
        # Gather the log probs for the target tokens
        gathered_log_probs = torch.gather(log_probs, -1, labels.unsqueeze(-1)).squeeze(-1)
        
        # Mask out padding tokens
        valid_mask = (labels != self.tokenizer.pad_token_id).float()
        masked_log_probs = gathered_log_probs * valid_mask
        
        return masked_log_probs, valid_mask
    
    def save(self, path: str):
        """Save the policy model."""
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        
    def update(self, optimizer, loss):
        """Update the policy model using the provided loss."""
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
