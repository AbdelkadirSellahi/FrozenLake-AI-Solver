# Import necessary libraries
import gym
import numpy as np

# Initialize the FrozenLake environment from OpenAI Gym with human-readable rendering
env = gym.make("FrozenLake-v1", render_mode="human")

# Reset the environment to start a new episode
env.reset()

# Retrieve the total number of states from the environment's observation space
num_states = env.observation_space.n

# Retrieve the total number of actions from the environment's action space
num_action = env.action_space.n

# Define the discount factor for future rewards
gamma = 0.9

# Initialize the value function for all states to zero
V = np.zeros(num_states)

# Initialize the policy array, setting an action for each state to zero initially
policy = np.zeros(num_states, dtype=int)

# Main loop for the Value Iteration algorithm
for _ in range(1000):  # Limit the number of iterations to prevent infinite loops

    prev_V = np.copy(V)  # Copy the value function to check for convergence later
    # Loop through each state in the environment
    for s in range(num_states):

        q = np.zeros(num_action)  # Initialize a temporary array to store Q-values
        # Loop through each action available in the current state
        for a in range(num_action):

            # Access the transition probabilities, next state, reward, and done flag
            # for the current state-action pair from the environment's dynamics (P)
            for probability, next_state, reward, done in env.P[s][a]:

                # Update the Q-value for the action based on the Bellman equation
                q[a] += probability * (reward + gamma * prev_V[next_state])

        # Update the value function for the current state to the max Q-value
        V[s] = np.max(q)

        # Update the policy to take the action with the highest Q-value at current state
        policy[s] = np.argmax(q)

    # Check for convergence by comparing the change in the value function with a small threshold
    if np.max(np.abs(prev_V - V)) < 1e-3:
        break  # Exit the loop if the value function has converged

# Print the final policy in a grid format that matches the FrozenLake map
print(policy.reshape(4, 4))

# Demonstrate the application of the derived policy starting from the initial state
B = env.step(int(policy[0]))  # Take the first action according to the policy

# Continue taking actions according to the policy until reaching a terminal state
while not (B[2]):  # B[2] indicates whether the episode is done
    
    B = env.step(int(policy[B[0]]))  # B[0] is the next state from the previous action

# Close the environment after completion
env.close()