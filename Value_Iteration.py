# Import the necessary libraries
import gym
import numpy as np

# Initialize the FrozenLake environment with human-readable rendering
env = gym.make("FrozenLake-v1", render_mode="human")

# Reset the environment to start a new episode
env.reset()

# Render the initial state of the environment
env.render()

# Retrieve the total number of states from the environment's observation space
num_states = env.observation_space.n

# Retrieve the total number of possible actions from the environment's action space
num_action = env.action_space.n

# Set the discount factor for future rewards
gamma = 0.9

# Set the threshold for stopping the iteration based on value function convergence
epsilon = 10**-3

# Initialize the old value function with ones (arbitrary non-zero to start the loop)
V_old = np.ones(num_states)

# Initialize the current value function with zeros
V = np.zeros(num_states)

# Initialize the policy with zeros (arbitrary starting policy)
policy = np.zeros(num_states)

# Iterate until the maximum change in the value function is less than epsilon (convergence criterion)
while (np.abs(V - V_old).max() > epsilon):

    V_old = V.copy()  # Update the old value function for convergence check

    # Iterate through all states
    for s in range(num_states):

        q = np.zeros(num_action)  # Initialize a temporary array to store action values (Q-values)

        # Iterate through all possible actions for the current state
        for a in range(num_action):

            # Calculate the expected value of taking action a in state s
            for pr, st, rd, done in env.P[s][a]:

                q[a] += pr * (rd + gamma * V_old[st])  # Update Q-value based on Bellman equation

        V[s] = max(q)  # Update the value function to the maximum Q-value for state s

        policy[s] = np.argmax(q)  # Update the policy to choose the action with maximum Q-value

# Reshape and print the final policy in a grid format for visualization
print(policy.reshape(4,4))

# Simulate the environment using the derived optimal policy starting from the initial state
B = env.step(int(policy[0]))  # Take the first action according to the derived policy

# Continue taking steps according to the policy until the episode ends (reaching the goal or falling into a hole)
while not(B[2]):  # B[2] is the 'done' flag indicating whether the episode has ended
    
    B = env.step(int(policy[B[0]]))  # Select the next action based on the current state and the derived policy

# Close the environment after the simulation
env.close()