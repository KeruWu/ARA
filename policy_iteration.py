import numpy as np
from utility import Reward
from transition import trans_prob, new_w, h_normalize
from State import *

def All_states(Case_Study = 'Airport Security'):
    """
    Enumerating all possible states.
    Args:

    Returns:
        l: List of all states.
    """
    if Case_Study == 'Airport Security':

        l = []

        qs = Op_conditions()
        rs = Resources()

        ws = []
        w1 = Categorical_vectors(6)
        w2 = Categorical_vectors(5)
        w_iter= itertools.product(w1, w2)
        for i1, i2 in w_iter:
            ws.append(np.concatenate([i1, i2]))

        ITER = itertools.product(qs, rs, ws)
        for i1, i2, i3 in ITER:
            l.append([i1, i2, i3])

        return l
    else:
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


def same(States, policy1, policy2):
    """
    Check whether policy1 = policy2
    Args:
        States: List of all possible states.
        policy1: The first policy
        policy2: The second policy
    Returns:
        Logical True or False.
    """
    for s in States:
        if policy1(s) != policy2(s):
            return False
    return True

def Policy_iteration(States, m, tau, c, rho_da, rho_dq, h_above, h_below, g_above, g_below, dict_r,
                     gamma = 0.1, order= 0, max_iter = 100, seed = 17):
    """
    Function for policy iteration.
    Args:
        States: List of all possible states.
        m: Number of non multi-period commitments. (i.e. The first m defender's actions are not multi-period)
        tau: An array denoting the length of each multi-period commitment.
        c (nr * nd): cost of defender's each action
        rho_da: A map mapping from (d_i, a_j) to response quality
        rho_dq: A map mapping from (d_i, q) to response quality
        h_above: attraction value when response quality is above threshold
        h_below: attraction value when response quality is below threshold
        g_above: attraction value when response quality is above threshold
        g_below: attraction value when response quality is below threshold
        dict_r: map resource to corresponding level.

        gamma: Discount factor
        max_iter: Maximum number of iterations
        order: Order of ARA. Currently only 0 and 1 are available.
        seed: Random seed
    Returns:
        pi: Optimal policy given by policy iteration.
    """
    S = len(States)
    pi = Policy_initializer(seed)


    for iter in range(max_iter):

        prev_pi = pi
        if iter > 0 and same(States, prev_pi, pi):
            break

        ## Reward array
        R = np.zeros(S)
        for i, s in enumerate(States):
            R[i] = Reward(pi(s), s, order)

        ## Transition probability matrix
        P = np.zeros(S, S)
        for j, s in enumerate(States):
            next_w = new_w(pi(s), m, s, tau)
            #dict_h = h_normalize(next_w, pi(s), s, c)
            for i, next_s in enumerate(States):
                if np.all(next_s[2] == next_w):
                    P[i, j] = trans_prob(next_s, pi(s), s, m, tau, c,
                                         rho_da, rho_dq, h_above, h_below, g_above, g_below, dict_r) #, dict_h)

        ## Policy evaluation
        V = np.matmul(np.linalg.inv(np.identity(S)-gamma*P), R)

        ## Policy improvment
        def pi(s):
            best_V = 0
            best_d = 0
            for d in K(s, c):
                V_curr = Reward(prev_pi(s), s, order)
                ## the following iteration can be improved
                next_w = new_w(d, m, s, tau)
                #dict_h = h_normalize(next_w, d, s, c)
                for s_cand in States:
                    if np.all(s_cand[2] == next_w):
                        V_curr += gamma*trans_prob(s_cand, d, s, m, tau, c,
                                                   rho_da, rho_dq, h_above, h_below, g_above, g_below, dict_r) #, dict_h)
                if V_curr > best_V:
                    best_d = d
            return best_d
    return pi



def main():
    States = All_states()
    c1 = np.array([[0, 50],
                   [5*100, 0],
                   [10*100, 0],
                   [15*100, 0],
                   [20*100, 0],
                   [0, 120],
                   [0, 120],
                   [0, 120],
                   [0, 5*100],
                   [0, 10*100],
                   [0, 1000],
                   [0, 750]]).T
    c2 = np.array([[0, 50],
                   [5*100, 0],
                   [10*100, 0],
                   [15*100, 0],
                   [20*100, 0],
                   [0, 120],
                   [0, 120],
                   [0, 120],
                   [0, 5*100],
                   [0, 10*100],
                   [0, 60],
                   [0, 40]]).T
    c = [c1, c2]

    rho_da = np.array([[1, 1, 1],
                       [-1, -1, -0.5],
                       [0, 0, 0],
                       [0.5, 0.5, 1],
                       [1, 1, 0.5],
                       [1, 1, 0.5],
                       [1, 1, 0.5],
                       [1, 1, 0.5],
                       [0, 0, 1],
                       [1, 1, 0.5],
                       [1, 1, 1],
                       [1, 1, 1]])

    rho_dq = np.array([[1, 1, 1, 1, 1],
                       [1, 0.75, 0.5, 0.25, 0],
                       [0.75, 1, 0.75, 0.5, 0.25],
                       [0.5, 0.75, 1, 0.75, 0.5],
                       [0.25, 0.5, 0.75, 1, 0.75],
                       [1, 1, 1, 1, 1],
                       [1, 1, 1, 1, 1],
                       [1, 1, 1, 1, 1],
                       [1, 1, 0.5, 0, 0],
                       [0, 0, 0.5, 1, 1],
                       [1, 1, 1, 1, 1],
                       [1, 1, 1, 1, 1]])

    h_above = np.array([[0.4, 0.6, 0],
                        [0.2, 0.4, 0.4],
                        [0, 0.2, 0.8]])

    h_below = np.array([[0.9, 0.1, 0],
                        [0.4, 0.5, 0.1],
                        [0, 0.5, 0.5]])

    g_above = np.array([[0.6, 0.4, 0, 0, 0],
                        [0.3, 0.6, 0.1, 0, 0],
                        [0, 0.3, 0.6, 0.1, 0],
                        [0, 0, 0.3, 0.6, 0.1],
                        [0, 0, 0, 0.4, 0.6]])

    g_below = np.array([[0.4, 0.6, 0, 0, 0],
                        [0.2, 0.4, 0.4, 0, 0],
                        [0, 0.2, 0.4, 0.4, 0],
                        [0, 0, 0.2, 0.4, 0.4],
                        [0, 0, 0, 0.2, 0.8]])

    dict_r1 = {1000:0, 2000:1, 3000:2}
    dict_r2 = {50000:0, 75000:1, 100000:2}
    dict_r = [dict_r1, dict_r2]


    MDP = Policy_iteration(States, m=2, tau=np.array([5, 4]), c = c, rho_da = rho_da, rho_dq = rho_dq,
                           h_above=h_above, h_below=h_below, g_above=g_above, g_below=g_below,
                           dict_r=dict_r)

