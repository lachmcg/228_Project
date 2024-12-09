## Greedy Exploration ## 


from collections import defaultdict
from scipy.stats import dirichlet, norm
import pandas as pd


# Initialize parameters
alpha = 1  # Dirichlet prior parameter (pseudocount)
reward_std = 1  # Standard deviation for reward sampling
num_episodes = 100  # Number of episodes to run
max_steps = 200  # Maximum steps per episode
goal_reward = 100  # Large positive reward for reaching the goal

# Define the goal region (e.g., near the prominent hill's peak)
GOAL_X_COORD, GOAL_Y_COORD = prominent_hill_x, prominent_hill_y
goal_radius = 5  # Radius around the goal peak to consider as the goal region

# Sparse dictionary for transition counts and rewards
transition_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
reward_sums = defaultdict(lambda: defaultdict(float))
reward_counts = defaultdict(lambda: defaultdict(int))

# Data collection list
data = []

def get_reward(s, s_prime):
    x, y = s
    x_prime, y_prime = s_prime

    # Elevation change
    current_z_idx = state_map[x, y, 2]
    next_z_idx = state_map[x_prime, y_prime, 2]
    elevation_change = next_z_idx - current_z_idx

    # Distance traveled in the x-y plane
    distance_penalty = np.sqrt((x_prime - x)**2 + (y_prime - y)**2)

    # Compute distance to the goal for current and next states
    current_distance_to_goal = np.sqrt((x - GOAL_X_COORD)**2 + (y - GOAL_Y_COORD)**2)
    next_distance_to_goal = np.sqrt((x_prime - GOAL_X_COORD)**2 + (y_prime - GOAL_Y_COORD)**2)

    # Check if the next state is within the goal region
    if next_distance_to_goal <= goal_radius:
        return goal_reward

    # Potential-based shaping: reward is based on moving closer to the goal
    potential_difference = current_distance_to_goal - next_distance_to_goal
    reward = potential_difference - distance_penalty

    # Optionally, add a small incentive for uphill movement only when moving towards the goal
    if next_distance_to_goal < current_distance_to_goal:
        reward += 0.1 * elevation_change

    return reward


def deterministic_transition(s, a):
    """
    Determines the next state based on a deterministic action model with boundary handling.
    """
    x, y = s
    dx, dy = a
    new_x, new_y = x + dx, y + dy

    # Check if the new state is out of bounds
    if not (0 <= new_x < state_map.shape[0] and 0 <= new_y < state_map.shape[1]):
        return None  # Indicate an invalid transition
    return (new_x, new_y)

def posterior_sample_action(state):
    """
    Samples an action based on the posterior transition model using Thompson Sampling,
    and ensures the action does not lead out of bounds.
    """
    valid_action_samples = []
    for action in A:
        action_idx = A.index(action)

        # Check if the action would result in a valid next state
        next_state = deterministic_transition(state, action)
        if next_state is None:
            continue  # Skip invalid actions

        # Estimate expected reward for the action
        if reward_counts[state][action_idx] > 0:
            expected_reward = reward_sums[state][action_idx] / reward_counts[state][action_idx]
        else:
            expected_reward = -1  # Default penalty for unexplored actions

        valid_action_samples.append((action, expected_reward))

    # Choose the best valid action based on the highest expected reward
    if valid_action_samples:
        best_action = max(valid_action_samples, key=lambda x: x[1])[0]
        return best_action

    # Fallback to a random valid action if no valid actions are found
    return random.choice(A)

def simulate_episode(initial_state):
    """
    Simulates one episode using deterministic transitions with boundary handling.
    """
    state = initial_state
    trajectory = []
    for _ in range(max_steps):
        action = posterior_sample_action(state)
        action_idx = A.index(action)

        # Determine the next state using the deterministic transition model
        next_state = deterministic_transition(state, action)

        # If the transition was invalid, sample a new action
        if next_state is None:
            continue

        # Compute the reward for this transition
        reward = get_reward(state, next_state)

        # Store the transition data
        trajectory.append((state, action, reward, next_state))
        data.append((state, action, reward, next_state))

        # Update counts for posterior
        transition_counts[state][action_idx][next_state] += 1
        reward_sums[state][action_idx] += reward
        reward_counts[state][action_idx] += 1

        # Move to the next state
        state = next_state

    return trajectory


# Run simulations to collect exploration data
for episode in range(num_episodes):
    initial_state = random.choice(S)
    simulate_episode(initial_state)

# Flatten the data for CSV export
flattened_data = [
    [s[0], s[1], a[0], a[1], r, s_prime[0], s_prime[1]]
    for (s, a, r, s_prime) in data
]

# Create a DataFrame with explicit column names
df = pd.DataFrame(flattened_data, columns=["s_x", "s_y", "a_dx", "a_dy", "reward", "s'_x", "s'_y"])

# Save the DataFrame to CSV without quotes
df.to_csv("exploration_data_flat.csv", index=False)
print("Exploration data saved to exploration_data_flat.csv")

