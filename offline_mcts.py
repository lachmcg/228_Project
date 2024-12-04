## offline MCTS ## 

policy_table = {}

def extract_policy_with_mcts(mcts, all_states, max_depth=10):
    """
    Uses MCTS rollouts to extract a policy for all states in the state space.
    """
    for s in all_states:
        policy_table[s] = mcts(s)

    return policy_table

# Extract the policy offline
all_states = hill_climb_mdp.S
policy = extract_policy_with_mcts(mcts, all_states)

# Example of using the extracted policy
current_state = initial_state
while current_state not in policy_table or np.sqrt((current_state[0] - goal_x)**2 + (current_state[1] - goal_y)**2) > goal_radius:
    action = policy_table.get(current_state, random.choice(hill_climb_mdp.A))
    next_state = deterministic_transition(current_state, action)
    print(f"State: {current_state}, Action: {action}, Next State: {next_state}")
    if next_state is None:
        break
    current_state = next_state
