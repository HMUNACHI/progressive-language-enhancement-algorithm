import numpy as np


def value_iteration(env, gamma, theta, max_iterations, value=None):
    if value is None:
        value = np.zeros(env.n_states)
    else:
        value = np.array(value, dtype=np.float)
    
    for _ in range(max_iterations):
        delta = 0.
        
        for s in range(env.n_states):
            v = value[s]
            value[s] = max([sum([env.p(next_s, s, a) * (env.r(next_s, s, a) + gamma * value[next_s]) for next_s in range(env.n_states)]) for a in range(env.n_actions)])
    
            delta = max(delta, np.abs(v - value[s]))

        if delta < theta:
            break

    policy = np.zeros(env.n_states, dtype=int)
    for s in range(env.n_states):
        policy[s] = np.argmax([sum([env.p(next_s, s, a) * (env.r(next_s, s, a) + gamma * value[next_s]) for next_s in range(env.n_states)]) for a in range(env.n_actions)])

    return policy, value


def policy_evaluation(env, policy, gamma, theta, max_iterations):
    value = np.zeros(env.n_states, dtype=np.float)

    for _ in range(max_iterations):
        delta = 0
        for s in range(env.n_states):
            v = value[s]
            value[s] = sum([env.p(next_s, s, policy[s]) * (env.r(next_s, s, policy[s]) + gamma * value[next_s]) for next_s in range(env.n_states)])

            delta = max(delta, abs(v - value[s]))

        if delta < theta:
            break

    return value
