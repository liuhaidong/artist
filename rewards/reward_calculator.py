# rewards/reward_calculator.py
import re
from typing import Dict, Any, List, Tuple
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

class RewardCalculator:
    """
    Calculates rewards for ARTIST rollouts based on:
    1. Answer correctness
    2. Tool usage success
    3. Response formatting
    """
    
    def __init__(self, config):
        """
        Initialize the reward calculator.
        
        Args:
            config: Configuration object with reward parameters
        """
        self.config = config
        # Initialize sentence transformer for semantic similarity
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        
    def calculate_reward(self, 
                        rollout: Dict[str, Any], 
                        ground_truth: str) -> Dict[str, float]:
        """
        Calculate the reward for a rollout.
        
        Args:
            rollout: Rollout data containing response and tool usage
            ground_truth: Ground truth answer
            
        Returns:
            Dictionary of reward components and total reward
        """
        # Calculate answer correctness reward
        answer_correct, answer_reward = self._evaluate_answer(rollout["final_answer"], ground_truth)
        
        # Calculate tool usage reward
        tool_reward = self._evaluate_tool_usage(rollout)
        
        # Calculate formatting reward
        format_reward = self._evaluate_formatting(rollout["full_response"])
        
        # Combine rewards
        total_reward = (
            answer_reward + 
            self.config.tool_reward_weight * tool_reward +
            self.config.format_reward_weight * format_reward
        )
        
        return {
            "answer_correct": answer_correct,
            "answer_reward": answer_reward,
            "tool_reward": tool_reward,
            "format_reward": format_reward,
            "total_reward": total_reward,
            "tool_success_rate": rollout["tool_success_count"] / max(1, rollout["tool_total_count"])
        }
    
    def _evaluate_answer(self, 
                        predicted_answer: str, 
                        ground_truth: str) -> Tuple[bool, float]:
        """
        Evaluate the correctness of the answer.
        
        Args:
            predicted_answer: The model's answer
            ground_truth: The ground truth answer
            
        Returns:
            Tuple of (is_correct, reward_value)
        """
        # Normalize answers
        pred_norm = self._normalize_answer(predicted_answer)
        gt_norm = self._normalize_answer(ground_truth)
        
        # Exact match
        if pred_norm == gt_norm:
            return True, self.config.answer_reward
        
        # Check for numeric answers
        if self._is_numeric(pred_norm) and self._is_numeric(gt_norm):
            try:
                pred_num = float(pred_norm)
                gt_num = float(gt_norm)
                # Allow small relative error for numeric answers
                if abs(pred_num - gt_num) / (abs(gt_num) + 1e-10) < 0.01:
                    return True, self.config.answer_reward
            except:
                pass
        
        # Semantic similarity for text answers
        if len(pred_norm) > 0 and len(gt_norm) > 0:
            try:
                pred_embedding = self.sentence_model.encode([pred_norm])[0]
                gt_embedding = self.sentence_model.encode([gt_norm])[0]
                similarity = cosine_similarity([pred_embedding], [gt_embedding])[0][0]
                
                # High similarity gets partial credit
                if similarity > 0.9:
                    return True, self.config.answer_reward
                elif similarity > 0.8:
                    return False, 0.7 * self.config.answer_reward
                elif similarity > 0.7:
                    return False, 0.4 * self.config.answer_reward
            except:
                pass
        
        return False, 0.0
    
    def _evaluate_tool_usage(self, rollout: Dict[str, Any]) -> float:
        """
        Evaluate the tool usage in the rollout.
        
        Args:
            rollout: Rollout data containing tool usage information
            
        Returns:
            Tool usage reward
        """
        # No tools used
        if rollout["tool_total_count"] == 0:
            return 0.0
        
        # Calculate success rate
        success_rate = rollout["tool_success_count"] / rollout["tool_total_count"]
        
        # Reward based on success rate and number of tools used
        base_reward = success_rate
        
        # Encourage appropriate tool usage (not too many, not too few)
        tool_count_factor = min(1.0, 2.0 / max(1, rollout["tool_total_count"]))
        
        return base_reward * tool_count_factor
    
    def _evaluate_formatting(self, response: str) -> float:
        """
        Evaluate the formatting of the response.
        
        Args:
            response: The full response text
            
        Returns:
            Formatting reward
        """
        # Check for proper tool call format
        tool_format_correct = all(
            re.search(f"<{tool}>(.*?)</{tool}>", response, re.DOTALL) is not None
            for tool in re.findall(r"<(\w+)>", response)
            if tool != "output" and tool != "answer"
        )
        
        # Check for proper answer format
        answer_format_correct = "<answer>" in response and "</answer>" in response
        
        # Check for proper output format
        output_format_correct = all(
            re.search(r"<output>(.*?)</output>", response, re.DOTALL) is not None
            for _ in re.findall(r"<output>", response)
        )
        
        # Calculate formatting score
        format_score = (
            0.4 * tool_format_correct +
            0.4 * answer_format_correct +
            0.2 * output_format_correct
        )
        
        return format_score
    
    def _normalize_answer(self, text: str) -> str:
        """
        Normalize an answer string for comparison.
        
        Args:
            text: The text to normalize
            
        Returns:
            Normalized text
        """
        if text is None:
            return ""
            
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation and extra whitespace
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _is_numeric(self, text: str) -> bool:
        """
        Check if a string represents a numeric value.
        
        Args:
            text: The text to check
            
        Returns:
            True if numeric, False otherwise
        """
        try:
            float(text)
            return True
        except:
            return False
