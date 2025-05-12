# ARTIST Framework Minimal Implementation Demo
This is a minimal implementation of the ARTIST framework that demonstrates the core concepts of agentic reasoning with tool integration and reinforcement learning. This demo includes the essential components: policy model, tool execution, reward calculation, and GRPO optimization.

Agentic Reasoning and Tool Integration for LLMs via Reinforcement Learning
https://arxiv.org/abs/2505.01441

Project Structure
```
artist_demo/
├── models/
│   └── policy_model.py
├── tools/
│   └── tool_executor.py
├── rewards/
│   └── reward_calculator.py
├── rl/
│   └── grpo_optimizer.py
├── rollout/
│   └── rollout_engine.py
├── config.py
└── train.py
```

## How to Run

1. Install the required dependencies:
```bash
# Create a new conda environment named 'artist'
conda create -n artist python=3.11 -y

# Activate the environment
conda activate artist

pip install -r requirements.txt

```

2. Create necessary directories:
```bash
mkdir -p artist_demo/checkpoints
```

3. Run the training script:
```bash
python -m artist_demo.train
```

## Key Implementation Features

1. **Structured Reasoning Format**: Uses `<think>`, `<tool>`, `<output>`, and `<answer>` tags to structure the model's reasoning process.

2. **Tool Integration**: Implements a tool executor that can run Python code and simulate search queries.

3. **Reward Mechanism**: Calculates rewards based on answer correctness, format adherence, and tool execution success.

4. **GRPO Optimization**: Implements Group Relative Policy Optimization with advantage calculation based on group performance.

5. **Loss Masking**: Excludes tool output tokens from loss computation to focus optimization on the model's reasoning.

This minimal implementation captures the core concepts of the ARTIST framework. In a production environment, you would need to expand this with more sophisticated tool integrations, better reward models, and more efficient training procedures.
