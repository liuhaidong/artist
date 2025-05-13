# rollout/rollout_engine.py
import re
from typing import Dict, List, Optional, Tuple, Any
import json

class RolloutEngine:
    def __init__(self, policy_model, tools, config):
        self.policy_model = policy_model
        self.tools = {tool.name: tool for tool in tools}
        self.config = config
        
    def execute_rollout(self, 
                       prompt: str, 
                       temperature: float = None, 
                       do_sample: bool = True) -> Dict[str, Any]:
        """
        Execute a complete rollout for a given prompt.
        
        Args:
            prompt: The initial prompt to start the rollout
            temperature: Temperature for generation
            do_sample: Whether to sample or use greedy decoding
            
        Returns:
            Dict containing the complete rollout information
        """
        full_response = ""
        tool_calls = []
        tool_outputs = []
        tool_success_count = 0
        tool_total_count = 0
        
        # Initial generation
        current_prompt = prompt
        
        while True:
            # Generate the next segment
            response = self.policy_model.generate(
                current_prompt, 
                temperature=temperature,
                do_sample=do_sample
            )
            full_response += response
            
            # Check if we've reached the answer
            if "</answer>" in response:
                break
                
            # Check for tool calls
            tool_match = re.search(r"<(\w+)>(.*?)</\1>", response, re.DOTALL)
            if tool_match and tool_match.group(1) in self.tools:
                tool_name = tool_match.group(1)
                tool_query = tool_match.group(2).strip()
                
                # Execute the tool
                tool_total_count += 1
                tool_output, success = self.tools[tool_name].execute(tool_query)
                if success:
                    tool_success_count += 1
                
                # Record the tool call and output
                tool_calls.append({"tool": tool_name, "query": tool_query})
                tool_outputs.append({"output": tool_output, "success": success})
                
                # Append the tool output to the prompt for the next iteration
                output_tag = f"<output>{tool_output}</output>"
                current_prompt = current_prompt + response + output_tag
            else:
                # No tool call, just continue with the response
                current_prompt = current_prompt + response
                
            # Safety check to prevent infinite loops
            if len(full_response) > 10000 or tool_total_count > 10:  # Arbitrary limits
                break
        
        # Extract the final answer
        answer_match = re.search(r"<answer>(.*?)</answer>", full_response, re.DOTALL)
        final_answer = answer_match.group(1).strip() if answer_match else ""
        
        return {
            "full_response": full_response,
            "final_answer": final_answer,
            "tool_calls": tool_calls,
            "tool_outputs": tool_outputs,
            "tool_success_count": tool_success_count,
            "tool_total_count": tool_total_count
        }
        
    def generate_multiple_rollouts(self, 
                                 prompt: str, 
                                 num_rollouts: int,
                                 temperature: float = None) -> List[Dict[str, Any]]:
        """
        Generate multiple rollouts for GRPO.
        
        Args:
            prompt: The initial prompt
            num_rollouts: Number of rollouts to generate
            temperature: Temperature for generation
            
        Returns:
            List of rollout data dictionaries
        """
        if temperature is None:
            temperature = self.config.temperature
            
        rollouts = []
        for _ in range(num_rollouts):
            # Use a slightly different temperature for each rollout to increase diversity
            temp_variation = temperature * (0.9 + 0.2 * random.random())
            rollout = self.execute_rollout(prompt, temperature=temp_variation, do_sample=True)
            rollouts.append(rollout)
        return rollouts
