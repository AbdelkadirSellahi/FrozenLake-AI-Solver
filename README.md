# FrozenLake AI Solver

This repository contains implementations of fundamental `reinforcement learning` algorithms, `Value Iteration` , `Q-Learning` and `Policy Iteration`, designed to solve the `FrozenLake-v1` environment from OpenAI Gym. The solutions demonstrate the power of dynamic programming in solving discrete, deterministic models of decision-making.

## Introduction

The `FrozenLake` environment is a grid world where an agent navigates a slippery lake from a starting point to a goal. The agent must avoid falling into holes and reach the goal safely. This problem exemplifies a Markov Decision Process (MDP), ideal for applying reinforcement learning algorithms to solve.

## Contents

- `Value_Iteration.py`: Implements the Value Iteration algorithm.
- `Q_Learninig.py`: Implements the Q_Learninig algorithm.
- `Policy_iteration.py`: Implements the Policy Iteration algorithm.

## Getting Started

### Prerequisites

Ensure Python 3.6+ is installed. Dependencies `gym` and `numpy` are required and can be installed via pip:

```sh
pip install gym numpy
```

### Installation

Clone this repository:

```sh
git clone https://github.com/AbdelkadirSellahi/FrozenLake-AI-Solver.git
cd FrozenLake-AI-Solver
```

### Running the Algorithms

- Execute Value Iteration:

```sh
python Value_Iteration.py
```
- Execute Q_Learninig:

```sh
python Q_Learninig.py
```

- Execute Policy Iteration:

```sh
python Policy_iteration.py
```


## Algorithm Overview

### Value Iteration

This dynamic programming algorithm iteratively updates state values based on the Bellman optimality equation until convergence. It evaluates the expected utility of all possible actions in a state and chooses the action that maximizes this utility. This process continues for all states until the value function stabilizes, leading to the derivation of the optimal policy.

#### Key Features:
- **Direct Policy Derivation**: Does not start with an initial policy but derives the optimal policy from the value function.
- **Efficiency**: Typically converges faster than policy iteration for many problem instances.
- **Implementation Simplicity**: Generally simpler to implement and understand.

### Policy Iteration

Policy Iteration alternates between evaluating a policy (Policy Evaluation) and improving it (Policy Improvement) until the policy converges to the optimum. It starts with an arbitrary policy, calculates its value function, and iteratively updates the policy based on the current value function.

#### Steps:
1. **Policy Evaluation**: Computes the value of the current policy by solving the system of linear equations representing the expected returns for each state.
2. **Policy Improvement**: Updates the policy by choosing actions that maximize the expected utility based on the current value function.
3. **Convergence**: The process repeats until the policy is stable (no changes between iterations).

#### Key Features:
- **Guaranteed Convergence**: Always converges to the optimal policy.
- **Optimal Policy Identification**: Efficiently identifies the optimal policy by explicitly calculating the value of each policy.

### Q-Learning

Q-Learning is a model-free reinforcement learning algorithm that seeks to find the best action to take given the current state. It operates by learning an action-value function that ultimately gives the expected utility of taking a given action in a given state and following the optimal policy thereafter.

#### Algorithm Details:
- **Model-Free**: Q-Learning does not require a model of the environment and can learn directly from experiences of interactions with the environment.
- **Learning the Q-Function**: The core of the algorithm involves updating the Q-values (action-value pairs) for each state-action combination using the Bellman equation.
- **Exploration vs. Exploitation**: Q-Learning balances between exploring new actions to find the best rewards and exploiting the known actions that give high rewards. This is often managed by a parameter such as epsilon in an epsilon-greedy strategy.

#### Key Features:
- **Flexibility**: Can be applied to any episodic or non-episodic task, and to any environment (discrete or continuous states and actions).
- **Convergence**: With proper hyperparameter settings (learning rate, discount factor, and policy for choosing actions), Q-Learning is guaranteed to converge towards the optimal action-value function.
- **Off-Policy**: Learns the optimal policy regardless of the agent's actions, allowing it to learn from both historical and generated data.

Q-Learning's flexibility and off-policy nature make it a powerful algorithm for a wide range of problems where the model of the environment is unknown or the state and action spaces are large.

## Configuration Table

Below are the configurations and hyperparameters used in the implementations of the Value Iteration, Policy Iteration, and Q-Learning algorithms for the FrozenLake-v1 environment.

| Parameter               | Description                                      | Value Iteration | Policy Iteration | Q-Learning |
|-------------------------|--------------------------------------------------|-----------------|------------------|------------|
| **Environment**         | The RL environment used                          | FrozenLake-v1   | FrozenLake-v1    | FrozenLake-v1 |
| **Discount Factor (γ)** | Discount factor for future rewards               | 0.9             | 0.9              | 0.9        |
| **Learning Rate (α)**   | Learning rate for Q-Learning                     | N/A             | N/A              | 0.1        |
| **Epsilon (ε)**         | Exploration rate for epsilon-greedy strategy     | N/A             | N/A              | 0.1        |
| **Maximum Iterations**  | Max number of iterations before stopping         | 1000            | 1000             | 1000      |
| **Convergence Threshold** | Threshold for determining convergence         | 1e-3            | 1e-3             | N/A        |
| **Reward**              | Reward structure of the environment              | Standard        | Standard         | Standard   |


### Hyperparameter Explanation

- **Environment**: The `FrozenLake-v1` from OpenAI Gym is a grid world where the agent aims to reach a goal without falling into holes.
- **Discount Factor (γ)**: Represents how much future rewards are discounted, making them worth less than immediate rewards.
- **Learning Rate (α)**: Only applicable to Q-Learning. Determines to what extent newly acquired information overrides old information.
- **Epsilon (ε)**: Used in Q-Learning to balance exploration (trying new actions) and exploitation (taking known actions that give high rewards).
- **Maximum Iterations**: The maximum number of iterations the algorithm will run to find the optimal policy. It's a stopping criterion to prevent infinite loops.
- **Convergence Threshold**: Used in Value Iteration and Policy Iteration to determine when the value function or policy has sufficiently converged to stop the iteration.
- **Reward**: Describes the reward structure of the environment, which in the case of `FrozenLake-v1`, includes receiving a reward when reaching the goal.

## Contributing

We welcome contributions to improve the algorithms or documentation.

## Authors

- [**ABDELKADIR Sellahi**](https://github.com/AbdelkadirSellahi)

## Acknowledgments

- Thanks to OpenAI Gym for the FrozenLake environment.
