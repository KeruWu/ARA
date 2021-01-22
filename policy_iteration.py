import numpy as np
from utility import Reward
from transition import trans_prob
from State import Defender_actions, K

def All_states():
    """
    Enumerating all possible states.
    Args:

    Returns:
        l: List of all states.
    """
    l = []
    return l


def Policy_initializer(seed = 17):
    """
    Randomly initialize a policy.
    Args:
        seed: Random seed
    Returns:
        pi: A function given input state s, outputs the action at state s
    """

    def pi(s):
        np.random.seed(seed)
        D_actions = Defender_actions(s)
        return np.random.choice(D_actions)
    return pi


def same(policy1, policy2):
    """
    Check whether policy1 = policy2
    Args:
        policy1: The first policy
        policy2: The second policy
    Returns:
        Logical True or False.
    """

    return False


def Policy_iteration(m, tau, c, gamma = 0.1, order= 0, max_iter = 100, seed = 17):
    """
    Function for policy iteration.
    Args:
        m: Number of non multi-period commitments. (i.e. The first m defender's actions are not multi-period)
        tau: An array denoting the length of each multi-period commitment.
        c (nr * nd): cost of defender's each action
        gamma: Discount factor
        max_iter: Maximum number of iterations
        order: Order of ARA. Currently only 0 and 1 are available.
        seed: Random seed
    Returns:
        pi: Optimal policy given by policy iteration.
    """
    States = All_states()
    S = len(States)
    pi = Policy_initializer(seed)


    for iter in range(max_iter):

        prev_pi = pi
        if iter > 0 and same(prev_pi, pi):
            break

        ## Reward array
        R = np.zeros(S)
        for i, s in enumerate(States):
            R[i] = Reward(pi(s), s, order)

        ## Transition probability matrix
        P = np.zeros(S, S)
        for i, next_s in enumerate(States):
            for j, s in enumerate(States):
                P[i, j] = trans_prob(next_s, pi(s), s, m, tau, c)

        ## Policy evaluation
        V = np.matmul(np.linalg.inv(np.identity(S)-gamma*P), R)

        ## Policy improvment
        def pi(s):
            best_V = 0
            best_d = 0
            for d in K(s, c):
                V_curr = Reward(prev_pi(s), s, order)
                ## the following iteration can be improved
                for s_cand in States:
                    V_curr += gamma*trans_prob(s_cand, d, s, m, tau, c)
                if V_curr > best_V:
                    best_d = d
            return best_d
    return pi