# config.py
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

@dataclass
class ARTISTConfig:
    # Model configuration
    model_name: str = "Qwen/Qwen2.5-7B"
    max_length: int = 2048
    temperature: float = 0.7
    
    # Training configuration
    learning_rate: float = 5e-6
    batch_size: int = 4
    num_epochs: int = 3
    group_size: int = 8  # Number of rollouts per question for GRPO
    
    # Reward configuration
    answer_reward: float = 2.0
    format_reward_weight: float = 0.5
    tool_reward_weight: float = 0.5
    
    # GRPO configuration
    clip_ratio: float = 0.2
    kl_coef: float = 0.1
    max_grad_norm: float = 1.0
    ppo_epochs: int = 4  # Number of optimization epochs per batch
    
    # Tool configuration
    available_tools: List[str] = None
    
    def __post_init__(self):
        if self.available_tools is None:
            self.available_tools = ["python", "search"]
