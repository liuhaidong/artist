# main.py
import torch
import argparse
from tqdm import tqdm
import json
import os
import random
import numpy as np
from collections import defaultdict

from config import ARTISTConfig
from models.policy_model import PolicyModel
from rollout.rollout_engine import RolloutEngine
from tools.python_executor import PythonExecutor
from tools.search_tool import SearchTool
from rewards.reward_calculator import RewardCalculator
from rl.grpo_optimizer import GRPOOptimizer
from prompts.prompt_templates import PromptTemplates

def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def parse_args():
    parser = argparse.ArgumentParser(description="Train ARTIST framework")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B", help="Model name")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-6, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--group_size", type=int, default=8, help="Group size for GRPO")
    parser.add_argument("--data_path", type=str, required=True, help="Path to dataset")
    parser.add_argument("--output_dir", type=str, default="./output", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--eval_steps", type=int, default=100, help="Evaluation frequency")
    return parser.parse_args()

def load_dataset(data_path):
    """Load dataset from file."""
    with open(data_path, 'r') as f:
        data = json.load(f)
    return data

def main():
    args = parse_args()
    set_seed(args.seed)
    
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
    full_dataset = load_dataset(args.data_path)
    
    # Split into train and validation
    random.shuffle(full_dataset)
    val_size = min(100, int(0.1 * len(full_dataset)))
    train_dataset = full_dataset[val_size:]
    val_dataset = full_dataset[:val_size]
    
    print(f"Training on {len(train_dataset)} examples, validating on {len(val_dataset)} examples")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Training metrics
    metrics = defaultdict(list)
    global_step = 0
    best_val_reward = 0
    
    # Training loop
    for epoch in range(config.num_epochs):
        print(f"Epoch {epoch+1}/{config.num_epochs}")
        
        # Shuffle training data
        random.shuffle(train_dataset)
        
        # Training
        policy_model.model.train()
        
        for batch_idx in tqdm(range(0, len(train_dataset), config.batch_size)):
            batch = train_dataset[batch_idx:batch_idx + config.batch_size]
            
            batch_metrics = defaultdict(list)
            
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
                    batch_metrics["rewards"].append(reward_info["total_reward"])
                    batch_metrics["answer_correct"].append(float(reward_info["answer_correct"]))
                    batch_metrics["tool_success"].append(reward_info["tool_success_rate"])
                
                # Optimize policy using GRPO
                loss_info = optimizer.optimize(rollouts, prompt)
                for k, v in loss_info.items():
                    batch_metrics[f"loss_{k}"].append(v)
                
                # Log the best rollout
                best_rollout = max(rollouts, key=lambda x: x["reward"]["total_reward"])
                batch_metrics["best_rewards"].append(best_rollout["reward"]["total_reward"])
            
            # Update global metrics
            for k, v in batch_metrics.items():
                metrics[k].append(sum(v) / len(v) if v else 0)
            
            global_step += 1
            
            # Periodic evaluation
            if global_step % args.eval_steps == 0:
                val_metrics = evaluate(
                    policy_model, 
                    rollout_engine, 
                    reward_calculator, 
                    val_dataset, 
                    config
                )
                
                print(f"Step {global_step} - Validation:")
                print(f"  Reward: {val_metrics['reward']:.4f}")
                print(f"  Accuracy: {val_metrics['accuracy']:.4f}")
                print(f"  Tool Success: {val_metrics['tool_success']:.4f}")
                
                # Save best model
                if val_metrics["reward"] > best_val_reward:
                    best_val_reward = val_metrics["reward"]
                    best_model_path = os.path.join(args.output_dir, "best_model")
                    policy_model.save(best_model_path)
                    print(f"  New best model saved to {best_model_path}")
                
                # Save metrics
                with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
                    json.dump(metrics, f, indent=2)
        
        # End of epoch
        print(f"Epoch {epoch+1} - Training metrics:")
        print(f"  Mean reward: {np.mean(metrics['best_rewards'][-len(train_dataset)//config.batch_size:]):.4f}")
        print(f"  Mean accuracy: {np.mean(metrics['answer_correct'][-len(train_dataset)//config.batch_size:]):.4f}")
        print(f"  Mean tool success: {np.mean(metrics['tool_success'][-len(train_dataset)//config.batch_size:]):.4f}")
        
        # Save checkpoint
        checkpoint_path = os.path.join(args.output_dir, f"checkpoint_epoch_{epoch+1}")
        policy_model.save(checkpoint_path)
    
    print("Training complete!")

def evaluate(policy_model, rollout_engine, reward_calculator, dataset, config):
    """Evaluate the model on a dataset."""
    policy_model.model.eval()
    
    metrics = defaultdict(list)
    
    # Use a smaller batch size for evaluation to save memory
    eval_batch_size = min(config.batch_size, 4)
    
    for batch_idx in range(0, len(dataset), eval_batch_size):
        batch = dataset[batch_idx:batch_idx + eval_batch_size]
        
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
            
            # For evaluation, we use a single rollout with lower temperature
            rollout = rollout_engine.execute_rollout(
                prompt, temperature=0.3, do_sample=True
            )
            
            # Calculate reward
            reward_info = reward_calculator.calculate_reward(rollout, ground_truth)
            
            # Record metrics
            metrics["reward"].append(reward_info["total_reward"])
            metrics["accuracy"].append(float(reward_info["answer_correct"]))
            metrics["tool_success"].append(reward_info["tool_success_rate"])
    
    # Compute averages
    return {
        "reward": np.mean(metrics["reward"]),
        "accuracy": np.mean(metrics["accuracy"]),
        "tool_success": np.mean(metrics["tool_success"])
    }

if __name__ == "__main__":
    main()
