# rl/grpo_optimizer.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any

class GRPOOptimizer:
    def __init__(self, policy_model, config):
        self.policy_model = policy_model
        self.config = config
        self.optimizer = torch.optim.Adam(
            self.policy_model.model.parameters(),
            lr=config.learning_rate
        )
        
    def optimize(self, rollouts: List[Dict[str, Any]], prompt: str):
        """
        Optimize the policy using Group Relative Policy Optimization.
        
        Args:
            rollouts: List of rollouts with their rewards
            prompt: The original prompt
        """
        # Sort rollouts by reward
        sorted_rollouts = sorted(rollouts, key=lambda x: x["reward"]["total_reward"], reverse=True)
        
        # Tokenize the prompt for reference
        prompt_tokens = self.policy_model.tokenizer(prompt, return_tensors="pt").to(self.policy_model.model.device)
        prompt_length = prompt_tokens.input_ids.shape[1]
        
        # Prepare batches for optimization
        all_input_ids = []
        all_attention_masks = []
        all_labels = []
        all_advantages = []
        all_masks = []  # For masking tool output tokens
        
        # Calculate advantages based on group ranking
        group_size = len(rollouts)
        for i, rollout in enumerate(sorted_rollouts):
            # Relative advantage based on position in sorted list
            advantage = 1.0 - (i / (group_size - 1)) if group_size > 1 else 1.0
            
            # Tokenize the full response
            full_response = prompt + rollout["full_response"]
            tokens = self.policy_model.tokenizer(full_response, return_tensors="pt").to(self.policy_model.model.device)
            
            # Create labels (shift input_ids right)
            labels = tokens.input_ids.clone()
            labels[:, :prompt_length] = -100  # Don't compute loss for prompt tokens
            
            # Create mask for tool output tokens
            mask = torch.ones_like(labels)
            
            # Find and mask tool output tokens
            for output in rollout["tool_outputs"]:
                output_text = f"<output>{output['output']}</output>"
                output_tokens = self.policy_model.tokenizer(output_text, add_special_tokens=False).input_ids
                
                # Find this sequence in the full tokens
                for i in range(len(tokens.input_ids[0]) - len(output_tokens) + 1):
                    if torch.all(tokens.input_ids[0, i:i+len(output_tokens)] == torch.tensor(output_tokens).to(tokens.input_ids.device)):
                        mask[0, i:i+len(output_tokens)] = 0
                        break
            
            all_input_ids.append(tokens.input_ids)
            all_attention_masks.append(tokens.attention_mask)
            all_labels.append(labels)
            all_advantages.append(torch.tensor([advantage]).to(self.policy_model.model.device))
            all_masks.append(mask)
        
        # Stack all tensors
        input_ids = torch.cat(all_input_ids, dim=0)
        attention_mask = torch.cat(all_attention_masks, dim=0)
        labels = torch.cat(all_labels, dim=0)
        advantages = torch.cat(all_advantages, dim=0)
        masks = torch.cat(all_masks, dim=0)
        
        # Compute old log probs
        with torch.no_grad():
            old_log_probs, valid_mask = self.policy_model.compute_logprobs(input_ids, attention_mask, labels)
        
        # Update step
        self.optimizer.zero_grad()
        
        # Forward pass with new policy
        outputs = self.policy_model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        logits = outputs.logits
        log_probs = F.log_softmax(logits, dim=-1)
        
        # Gather the log probs for the target tokens
        gathered_log_probs = torch.gather(log_probs, -1, labels.unsqueeze(-1)).squeeze(-1)
        
        # Apply valid mask and tool output mask
        final_mask = valid_mask * masks
        masked_log_probs = gathered_log_probs * final_mask
        masked_old_log_probs = old_log_probs * final_mask
        
        # Compute ratio and clipped ratio
        ratio = torch.exp(masked_log_probs - masked_old_log_probs.detach())
        clipped_ratio = torch.clamp(ratio, 1.0 - self.config.clip_ratio, 1.0 + self.config.clip_ratio)
        
        # Compute losses
        policy_loss = -torch.min(
            ratio * advantages.unsqueeze(-1),
            clipped_ratio * advantages.unsqueeze(-1)
        )
        
        # Apply mask and compute mean
        policy_loss = (policy_loss * final_mask).sum() / final_mask.sum()
        
        # Add KL penalty
        kl_div = (masked_old_log_probs - masked_log_probs) * final_mask
        kl_penalty = self.config.kl_coef * kl_div.mean()
        
        # Total loss
        loss = policy_loss + kl_penalty
        
        # Backward and optimize
        loss.backward()
        self.optimizer.step()
        
        return {
            "policy_loss": policy_loss.item(),
            "kl_penalty": kl_penalty.item(),
            "total_loss": loss.item()
        }
