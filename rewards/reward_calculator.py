# rewards/reward_calculator.py
import re
from typing import Dict, Any, List

class RewardCalculator:
    def __init__(self, config):
        self.config = config
        
    def calculate_reward(self, rollout: Dict[str, Any], ground_truth: str) -> Dict[str, float]:
        """
        Calculate rewards for a rollout.
        
        Args:
            rollout: The rollout data
            ground_truth: The correct answer
            
        Returns:
            Dict of reward components and total reward
        """
        # Calculate answer reward
        answer_correct = self._check_answer_correctness(rollout["final_answer"], ground_truth)
        answer_reward = self.config.answer_reward if answer_correct else 0.0
        
        # Calculate format reward
        format_correct = self._check_format_correctness(rollout["full_response"])
        format_reward = self.config.format_reward_weight if format_correct else 0.0
        
        # Calculate tool execution reward
        tool_success_rate = 0.0
        if rollout["tool_total_count"] > 0:
            tool_success_rate = rollout["tool_success_count"] / rollout["tool_total_count"]
        tool_reward = self.config.tool_reward_weight * tool_success_rate
        
        # Calculate total reward
        total_reward = answer_reward + format_reward + tool_reward
        
        return {
            "answer_reward": answer_reward,
            "format_reward": format_reward,
            "tool_reward": tool_reward,
            "total_reward": total_reward,
            "answer_correct": answer_correct,
            "format_correct": format_correct,
            "tool_success_rate": tool_success_rate
        }
    
    def _check_answer_correctness(self, predicted_answer: str, ground_truth: str) -> bool:
        """
        Check if the predicted answer is correct.
        
        In a real implementation, this would be more sophisticated and task-specific.
        """
        # Simple string matching for demonstration
        # In practice, you'd want more robust answer checking
        return predicted_answer.strip() == ground_truth.strip()
    
    def _check_format_correctness(self, full_response: str) -> bool:
        """Check if the response follows the correct format structure."""
        # Check for proper tag nesting and ordering
        think_tags = re.findall(r"<think>.*?</think>", full_response, re.DOTALL)
        tool_tags = re.findall(r"<\w+>.*?</\w+>", full_response, re.DOTALL)
        output_tags = re.findall(r"<output>.*?</output>", full_response, re.DOTALL)
        answer_tags = re.findall(r"<answer>.*?</answer>", full_response, re.DOTALL)
        
        # Basic checks
        has_answer = len(answer_tags) > 0
        proper_nesting = True  # Simplified check
        
        return has_answer and proper_nesting
