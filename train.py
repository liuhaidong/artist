def train_artist_with_grpo(policy_model, dataset, config):
    """
    Algorithm 1: Training ARTIST with Group Relative Policy Optimization
    
    Args:
        policy_model: The policy model to train
        dataset: Training dataset
        config: Configuration parameters
    """
    # Initialize components
    tools = initialize_tools(config.available_tools)
    rollout_engine = RolloutEngine(policy_model, tools, config)
    reward_calculator = RewardCalculator(config)
    optimizer = GRPOOptimizer(policy_model, config)
    
    # Training loop
    for iteration in range(config.num_iterations):
        # Sample batch of tasks
        batch = sample_batch(dataset, config.batch_size)
        
        for task in batch:
            question = task["question"]
            ground_truth = task["answer"]
            
            # Generate prompt
            prompt = generate_prompt(question, task.get("task_type"))
            
            # Sample G rollouts from current policy
            rollouts = rollout_engine.generate_multiple_rollouts(
                prompt, 
                num_rollouts=config.group_size
            )
            
            # Calculate rewards for each rollout
            for rollout in rollouts:
                reward_info = reward_calculator.calculate_reward(rollout, ground_truth)
                rollout["reward"] = reward_info
            
            # Optimize policy using GRPO
            optimizer.optimize(rollouts, prompt)
        
        # Periodically evaluate and save the model
        if iteration % config.eval_frequency == 0:
            evaluate_and_save(policy_model, dataset, config)
