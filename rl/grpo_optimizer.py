# rl/grpo_optimizer.py
import torch
import torch.nn.functional as F
import re
from typing import List, Dict, Any, Tuple

class GRPOOptimizer:
    """
    Implementation of Group Relative Policy Optimization (GRPO) for ARTIST framework.
    
    GRPO is a policy optimization algorithm that:
    1. Uses relative ranking within groups to compute advantages
    2. Applies importance sampling with clipping
    3. Masks tool output tokens during optimization
    4. Adds KL divergence regularization
    """
    
    def __init__(self, policy_model, config):
        """
        Initialize the GRPO optimizer.
        
        Args:
            policy_model: The policy model to optimize
            config: Configuration object with GRPO parameters
        """
        self.policy_model = policy_model
        self.config = config
        self.optimizer = torch.optim.Adam(
            self.policy_model.model.parameters(),
            lr=config.learning_rate
        )
        
    def create_token_masks(self, 
                          rollout: Dict[str, Any], 
                          tokenized_sequence: torch.Tensor) -> torch.Tensor:
        """
        Create masks for tool output tokens.
        
        Args:
            rollout: Rollout data containing tool outputs
            tokenized_sequence: Tokenized full sequence
            
        Returns:
            Tensor mask where 0 indicates tool output tokens to be masked
        """
        # Start with all tokens unmasked (1)
        mask = torch.ones_like(tokenized_sequence)
        
        # Find and mask all tool output tokens
        for tool_output in rollout["tool_outputs"]:
            output_text = f"<output>{tool_output['output']}</output>"
            
            # Find the start and end positions of this output in the full text
            full_text = rollout["full_response"]
            start_idx = full_text.find(output_text)
            
            if start_idx != -1:
                end_idx = start_idx + len(output_text)
                
                # Convert text positions to token positions (approximate)
                prefix_tokens = self.policy_model.tokenizer(
                    full_text[:start_idx], 
                    add_special_tokens=False, 
                    return_tensors="pt"
                ).input_ids
                
                output_tokens = self.policy_model.tokenizer(
                    output_text, 
                    add_special_tokens=False, 
                    return_tensors="pt"
                ).input_ids
                
                # Calculate token positions
                start_token = prefix_tokens.shape[1]
                end_token = start_token + output_tokens.shape[1]
                
                # Apply mask (accounting for prompt offset)
                prompt_length = tokenized_sequence.shape[1] - len(rollout["full_response"])
                mask[0, prompt_length + start_token:prompt_length + end_token] = 0
        
        return mask
    
    def extract_tool_output_spans(self, text: str) -> List[Tuple[int, int]]:
        """
        Extract the character spans of all tool outputs in the text.
        
        Args:
            text: The full response text
            
        Returns:
            List of (start, end) character positions for tool outputs
        """
        output_spans = []
        pattern = r"<output>(.*?)</output>"
        
        for match in re.finditer(pattern, text, re.DOTALL):
            output_spans.append((match.start(), match.end()))
            
        return output_spans
    
    def compute_token_level_masks(self, 
                                 full_text: str, 
                                 tokenized: torch.Tensor, 
                                 prompt_length: int) -> torch.Tensor:
        """
        Compute precise token-level masks for tool outputs.
        
        Args:
            full_text: The full text including prompt and response
            tokenized: The tokenized full sequence
            prompt_length: Number of tokens in the prompt
            
        Returns:
            Binary mask tensor (1 for tokens to keep, 0 for tokens to mask)
        """
        # Start with all tokens unmasked
        mask = torch.ones_like(tokenized.input_ids)
        
        # Get all output spans
        output_spans = self.extract_tool_output_spans(full_text)
        
        # For each span, find the corresponding tokens and mask them
        for start_char, end_char in output_spans:
            # Get the text before the output
            prefix_text = full_text[:start_char]
            
            # Tokenize the prefix to find token position
            prefix_tokens = self.policy_model.tokenizer(
                prefix_text, 
                add_special_tokens=False
            ).input_ids
            
            # Tokenize the output text
            output_text = full_text[start_char:end_char]
            output_tokens = self.policy_model.tokenizer(
                output_text, 
                add_special_tokens=False
            ).input_ids
            
            # Calculate token positions
            start_token = len(prefix_tokens)
            end_token = start_token + len(output_tokens)
            
            # Apply mask (accounting for prompt offset)
            mask[:, prompt_length + start_token:prompt_length + end_token] = 0
            
        return mask
    
    def compute_advantages(self, rewards: List[float]) -> torch.Tensor:
        """
        Compute advantages based on relative ranking within the group.
        
        Args:
            rewards: List of rewards for each rollout
            
        Returns:
            Tensor of advantage values
        """
        # Sort rewards in descending order
        sorted_indices = torch.argsort(torch.tensor(rewards), descending=True)
        
        # Create advantages based on rank
        group_size = len(rewards)
        advantages = torch.zeros(group_size)
        
        for rank, idx in enumerate(sorted_indices):
            # Linear advantage scaling based on rank
            advantages[idx] = 1.0 - (rank / (group_size - 1)) if group_size > 1 else 1.0
            
        return advantages
    
    def optimize(self, rollouts: List[Dict[str, Any]], prompt: str) -> Dict[str, float]:
        """
        Optimize the policy using Group Relative Policy Optimization.
        
        Args:
            rollouts: List of rollouts with their rewards
            prompt: The original prompt
            
        Returns:
            Dict of loss metrics
        """
        # Extract rewards
        rewards = [r["reward"]["total_reward"] for r in rollouts]
        
        # Compute advantages based on reward ranking
        advantages = self.compute_advantages(rewards)
        advantages = advantages.to(self.policy_model.model.device)
        
        # Tokenize the prompt for reference
        prompt_tokens = self.policy_model.tokenizer(
            prompt, 
            return_tensors="pt"
        ).to(self.policy_model.model.device)
        prompt_length = prompt_tokens.input_ids.shape[1]
        
        # Prepare batches for optimization
        all_input_ids = []
        all_attention_masks = []
        all_labels = []
        all_tool_masks = []  # For masking tool output tokens
        
        # Process each rollout
        for rollout in rollouts:
            # Tokenize the full sequence (prompt + response)
            full_text = prompt + rollout["full_response"]
            tokens = self.policy_model.tokenizer(
                full_text, 
                return_tensors="pt",
                padding="max_length",
                max_length=self.config.max_length,
                truncation=True
            ).to(self.policy_model.model.device)
            
            # Create labels (shift input_ids right for next-token prediction)
            labels = tokens.input_ids.clone()
            labels[:, :prompt_length] = -100  # Don't compute loss for prompt tokens
            
            # Create mask for tool output tokens
            tool_mask = self.compute_token_level_masks(
                full_text, 
                tokens, 
                prompt_length
            ).to(self.policy_model.model.device)
            
            all_input_ids.append(tokens.input_ids)
            all_attention_masks.append(tokens.attention_mask)
            all_labels.append(labels)
            all_tool_masks.append(tool_mask)
        
        # Stack all tensors
        input_ids = torch.cat(all_input_ids, dim=0)
        attention_mask = torch.cat(all_attention_masks, dim=0)
        labels = torch.cat(all_labels, dim=0)
        tool_masks = torch.cat(all_tool_masks, dim=0)
        
        # Compute old log probs (from current policy before update)
        with torch.no_grad():
            outputs = self.policy_model.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            logits = outputs.logits
            old_log_probs = F.log_softmax(logits, dim=-1)
            
            # Gather the log probs for the target tokens
            old_token_log_probs = torch.gather(
                old_log_probs, 
                dim=2, 
                index=labels.unsqueeze(-1)
            ).squeeze(-1)
            
            # Create valid token mask (exclude padding and -100 label tokens)
            valid_token_mask = (labels != -100) & (labels != self.policy_model.tokenizer.pad_token_id)
            valid_token_mask = valid_token_mask.float()
            
            # Apply tool output mask
            final_mask = valid_token_mask * tool_masks
        
        # GRPO update
        for _ in range(self.config.ppo_epochs):
            # Forward pass with current policy
            outputs = self.policy_model.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            logits = outputs.logits
            log_probs = F.log_softmax(logits, dim=-1)
            
            # Gather log probs of chosen tokens
            token_log_probs = torch.gather(
                log_probs, 
                dim=2, 
                index=labels.unsqueeze(-1)
            ).squeeze(-1)
            
            # Compute importance ratio
            ratio = torch.exp(token_log_probs - old_token_log_probs)
            
            # Clipped ratio
            clipped_ratio = torch.clamp(
                ratio, 
                1.0 - self.config.clip_ratio, 
                1.0 + self.config.clip_ratio
            )
            
            # Compute token-level surrogate losses
            surrogate1 = ratio * advantages.unsqueeze(-1).expand_as(ratio)
            surrogate2 = clipped_ratio * advantages.unsqueeze(-1).expand_as(ratio)
            policy_loss = -torch.min(surrogate1, surrogate2)
            
            # Apply final mask and compute mean
            masked_policy_loss = (policy_loss * final_mask)
            policy_loss = masked_policy_loss.sum() / (final_mask.sum() + 1e-8)
            
            # KL divergence penalty
            kl_div = (old_token_log_probs - token_log_probs) * final_mask
            kl_penalty = self.config.kl_coef * kl_div.sum() / (final_mask.sum() + 1e-8)
            
            # Total loss
            loss = policy_loss + kl_penalty
            
            # Backward and optimize
            self.optimizer.zero_grad()
            loss.backward()
            
            # Apply gradient clipping
            if self.config.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.policy_model.model.parameters(), 
                    self.config.max_grad_norm
                )
                
            self.optimizer.step()
        
        return {
            "policy_loss": policy_loss.item(),
            "kl_penalty": kl_penalty.item(),
            "total_loss": loss.item(),
            "mean_ratio": (ratio * final_mask).sum() / (final_mask.sum() + 1e-8),
            "mean_advantage": advantages.mean().item()
        }
