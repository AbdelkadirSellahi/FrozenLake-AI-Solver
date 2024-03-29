# Import necessary libraries: gym for the environment and numpy for numerical operations
import gym
import numpy as np

# Initialize the FrozenLake environment with human-readable rendering
env = gym.make("FrozenLake-v1", render_mode="human")

# Reset the environment to its initial state
env.reset()

# Render the initial state of the environment for visualization
env.render()

# Retrieve the number of possible states in the environment
num_state = env.observation_space.n

# Retrieve the number of possible actions in the environment
num_action = env.action_space.n

# Set the discount factor for future rewards
gamma = 0.9

# Initialize a policy with an arbitrary action (0) for each state
policy = np.zeros(num_state)

# Initialize the old value function to -1 for all states (ensures the while loop starts)
v_old = -1 * np.ones(num_state)

# Begin Policy Iteration process
v = np.zeros(num_state)  # Initialize the current value function to zeros
for i in range(1000):  # Limit the number of iterations for policy iteration

    # Policy Evaluation step
    v_old = -1 * np.ones(num_state)  # Reset old value function to ensure the loop condition is met

    while (np.abs(v - v_old)).max() > 1e-3:  # Continue until value function convergence
        v_old = v.copy()  # Update the old value function

        for s in range(num_state):  # For each state in the environment

            vs = 0  # Temporary variable for storing the value of state s

            # Calculate the expected value of the current policy
            for prob, st, r, done in env.P[s][policy[s]]:
                vs += prob * (r + gamma * v[st])  # Bellman equation for policy evaluation
            v[s] = vs  # Update the value of state s

        print(v)  # Print the current value function for debugging/observation

    # Policy Improvement step
    for s in range(num_state):  # For each state in the environment

        q = np.zeros(num_action)  # Initialize a temporary Q-value array for all actions

        # For each action, calculate its Q-value
        for a in range(num_action):

            for prob, st, r, done in env.P[s][a]:
                q[a] += prob * (r + gamma * v[st])  # Bellman equation for Q-value

        policy[s] = np.argmax(q)  # Update the policy to choose the action with the highest Q-value

# Reshape and print the optimal policy and value function for better readability
print("Optimal Policy:")
print(policy.reshape(4, 4))
print("Optimal Value Function:")
print(v)

# Simulate the environment using the derived optimal policy
B = env.step(int(policy[0]))  # Take the first action according to the optimal policy

while not(B[2]):  # Continue taking steps until the episode ends

    B = env.step(int(policy[B[0]]))  # Select the next action based on the current state and optimal policy
    env.render()  # Render the current state of the environment
    
env.close()  # Close the environment after the simulation ends