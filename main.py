# main.py
import torch
import argparse
from tqdm import tqdm
import json
import os

from config import ARTISTConfig
from models.policy_model import PolicyModel
from rollout.rollout_engine import RolloutEngine
from tools.python_executor import PythonExecutor
from tools.search_tool import SearchTool
from rewards.reward_calculator import RewardCalculator
from rl.grpo_optimizer import GRPOOptimizer
from prompts.prompt_templates import PromptTemplates

def parse_args():
    parser = argparse.ArgumentParser(description="Train ARTIST framework")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B", help="Model name")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-6, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--group_size", type=int, default=8, help="Group size for GRPO")
    parser.add_argument("--data_path", type=str, required=True, help="Path to dataset")
    parser.add_argument("--output_dir", type=str, default="./output", help="Output directory")
    return parser.parse_args()

def load_dataset(data_path):
    """Load dataset from file."""
    with open(data_path, 'r') as f:
        data = json.load(f)
    return data

def main():
    args = parse_args()
    
    # Create config
    config = ARTISTConfig(
        model_name=args.model,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        num_epochs=args.epochs,
        group_size=args.group_size
    )
    
    # Initialize components
    policy_model = PolicyModel(config)
    tools = [PythonExecutor(), SearchTool()]
    rollout_engine = RolloutEngine(policy_model, tools, config)
    reward_calculator = RewardCalculator(config)
    optimizer = GRPOOptimizer(policy_model, config)
    
    # Load dataset
    dataset = load_dataset(args.data_path)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Training loop
    for epoch in range(config.num_epochs):
        print(f"Epoch {epoch+1}/{config.num_epochs}")
        
        total_rewards = []
        total_losses = []
        
        for batch_idx in tqdm(range(0, len(dataset), config.batch_size)):
            batch = dataset[batch_idx:batch_idx + config.batch_size]
            
            batch_losses = []
            batch_rewards = []
            
            for item in batch:
                question = item["question"]
                ground_truth = item["answer"]
                
                # Generate prompt based on task type
                if item.get("task_type") == "function_calling":
                    prompt = PromptTemplates.get_function_calling_prompt(
                        question, config.available_tools
                    )
                else:  # Default to math reasoning
                    prompt = PromptTemplates.get_math_reasoning_prompt(question)
                
                # Generate multiple rollouts
                rollouts = rollout_engine.generate_multiple_rollouts(
                    prompt, config.group_size
                )
                
                # Calculate rewards for each rollout
                for rollout in rollouts:
                    reward_info = reward_calculator.calculate_reward(rollout, ground_truth)
                    rollout["reward"] = reward_info
                    batch_rewards.append(reward_info["total_reward"])
                
                # Optimize policy using GRPO
                loss_info = optimizer.optimize(rollouts, prompt)
                batch_losses.append(loss_info["total_loss"])
                
                # Log the best rollout
                best_rollout = max(rollouts, key=lambda x: x["reward"]["total_reward"])
                print(f"Question: {question}")
                print(f"Best answer: {best_rollout['final_answer']}")
                print(f"Ground truth: {ground_truth}")
                print(f"Reward: {best_rollout['reward']['total_reward']}")
                print("---")
            
            total_rewards.extend(batch_rewards)
            total_losses.extend(batch_losses)
            
        # End of epoch statistics
        avg_reward = sum(total_rewards) / len(total_rewards) if total_rewards else 0
        avg_loss = sum(total_losses) / len(total_losses) if total_losses else 0
        
        print(f"Epoch {epoch+1} - Avg reward: {avg_reward:.4f}, Avg loss: {avg_loss:.4f}")
        
        # Save checkpoint
        checkpoint_path = os.path.join(args.output_dir, f"checkpoint_epoch_{epoch+1}")
        policy_model.save(checkpoint_path)
        
    print("Training complete!")

if __name__ == "__main__":
    main()
