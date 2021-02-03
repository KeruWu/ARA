import numpy as np
from utility import Reward
from transition import trans_prob, new_w
from State import *
from tqdm import tqdm
import time

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


def All_states_w(next_w, Case_Study = 'Airport Security'):
    """
    Enumerating all possible states with w = next_w
    Args:

    Returns:
        next_w: Multi-period commitments in the next epoch.
        l: List of all states.
    """
    if Case_Study == 'Airport Security':

        l = []

        qs = Op_conditions()
        rs = Resources()

        ws = [next_w]

        ITER = itertools.product(qs, rs, ws)
        for i1, i2, i3 in ITER:
            l.append([i1, i2, i3])

        return l
    else:
        l = []
        return l



def Policy_initializer(States, D_actions, seed = 17):
    """
    Randomly initialize a policy.
    Args:
        States: List of all possible states.
        seed: Random seed
    Returns:
        pi: A function given input state s, outputs the action at state s
    """
    pi = {}
    for s in States:
        np.random.seed((seed + hash(str(s))) % (1 << 32))
        idx = np.random.choice(len(D_actions))
        pi[str(s)] = D_actions[idx]
    return pi

"""
    def pi(s):
        np.random.seed((seed+hash(str(s)))%(1<<32))
        D_actions = Defender_actions(s)
        idx = np.random.choice(len(D_actions))
        return D_actions[idx]
    return pi
"""


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
        if not np.all(policy1[str(s)] == policy2[str(s)]):
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
    d_actions = Defender_actions()
    pi = Policy_initializer(States, d_actions, seed)
    prev_pi = pi
    S_dict = {str(s):i for i, s in enumerate(States)}

    for iter in tqdm(range(max_iter)):

        print("iter %d" % iter)
        t1 = time.time()
        if iter > 0 and same(States, prev_pi, pi):
            break
        prev_pi = pi
        t2 = time.time()
        print('Checking same pi t = %.2f' % (t2-t1))

        ## Reward array
        R = np.zeros(S)
        for i, s in enumerate(States):
            R[i] = Reward(pi[str(s)], s, c, order)
        t3 = time.time()
        print('Calculating reward t = %.2f' % (t3 - t2))

        ## Transition probability matrix
        P = np.zeros((S, S))
        for j, s in tqdm(enumerate(States)):
            next_w = new_w(pi[str(s)], m, s, tau)
            d = pi[str(s)]
            #dict_h = h_normalize(next_w, pi(s), s, c)
            for i, next_s in enumerate(States):
                if np.all(next_s[2] == next_w):
                    P[i, j] = trans_prob(next_s, d, s, m, tau, c,
                                         rho_da, rho_dq, h_above, h_below, g_above, g_below, dict_r) #, dict_h)
        t4 = time.time()
        print('Calculating P t = %.2f' % (t4 - t3))

        ## Policy evaluation
        V = np.matmul(np.linalg.inv(np.identity(S)-gamma*P), R)
        t5 = time.time()
        print('Policy evaluation, t = %.2f' % (t5 - t4))

        ## Policy improvment
        #def pi(s):
        pi = {}
        for s in tqdm(States):
            best_V = -10000000
            best_d = 0
            for d in K(s, c, d_actions):
                V_curr = Reward(d, s, c, order)
                ## the following iteration can be improved
                next_w = new_w(d, m, s, tau)
                #dict_h = h_normalize(next_w, d, s, c)
                States_w = All_states_w(next_w)
                tmp = 0
                for s_cand in States_w:
                    #if np.all(s_cand[2] == next_w):
                    tmp += trans_prob(s_cand, d, s, m, tau, c,
                                               rho_da, rho_dq, h_above, h_below,
                                               g_above, g_below, dict_r) * V[S_dict[str(s_cand)]] #, dict_h)
                V_curr += tmp * gamma
                if V_curr > best_V:
                    best_V = V_curr
                    best_d = d
            pi[str(s)] = best_d
         #   return best_d
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


    MDP = Policy_iteration(States, m=10, tau=np.array([5, 4]), c = c, rho_da = rho_da, rho_dq = rho_dq,
                           h_above=h_above, h_below=h_below, g_above=g_above, g_below=g_below,
                           dict_r=dict_r)

    return MDP


if __name__ == '__main__':
    #print(Defender_actions())
    #print(len(All_states()))
    pi = main()
