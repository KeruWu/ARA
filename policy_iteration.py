import numpy as np
from utility import *
from transition import *
from State import *
from tqdm import tqdm
from scipy.stats import dirichlet, beta, expon
import time


def All_states():
    """
    Enumerating all possible states.
    Args:

    Returns:
        l: List of all states.
    """

    l = []

    qs = Op_conditions()
    rs = Resources()

    ws = [np.array([i, 0, 0]) for i in range(52)]
    ws += [np.array([0, i, 0]) for i in range(52)]
    ws += [np.array([0, 0, i]) for i in range(52)]

    ITER = itertools.product(qs, rs, ws)
    for i1, i2, i3 in ITER:
        l.append([i1, i2, i3])

    return l


def All_states_w(next_w):
    """
    Enumerating all possible states with w = next_w
    Args:

    Returns:
        next_w: Multi-period commitments in the next epoch.
        l: List of all states.
    """


    l = []

    qs = Op_conditions()
    rs = Resources()

    ws = [next_w]

    ITER = itertools.product(qs, rs, ws)
    for i1, i2, i3 in ITER:
        l.append([i1, i2, i3])

    return l


def Policy_initializer(States, c, d_actions, seed = 17):
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
        d_satis = K(s, c, d_actions)
        idx = np.random.choice(len(d_satis))
        pi[str(s)] = d_satis[idx]
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
    cnt = 0
    for s in States:
        if not np.all(policy1[str(s)] == policy2[str(s)]):
            cnt += 1
    print(len(States)-cnt)
    return cnt == 0



def Policy_iteration(States, c, gamma = 0.1, max_iter = 100, seed = 17):
    """
    Function for policy iteration.
    Args:
        States: List of all possible states.
        c (nr * nd): cost of defender's each action
        gamma: Discount factor
        max_iter: Maximum number of iterations
        seed: Random seed
    Returns:
        pi: Optimal policy given by policy iteration.
    """
    S = len(States)
    d_actions = Defender_actions()
    pi = Policy_initializer(States, c, d_actions, seed=seed)
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
            R[i] = Reward(pi[str(s)], s, c)
        t3 = time.time()
        print('Calculating reward t = %.2f' % (t3 - t2))

        ## Transition probability matrix
        P = np.zeros((S, S))
        for i, s in enumerate(States):
            d = pi[str(s)]
            next_w = new_w(s[2], d)
            for j, next_s in enumerate(States):
                if np.all(next_s[2] == next_w):
                    P[i, j] = trans_prob(next_s, s[0][0], d)
        t4 = time.time()
        print('Calculating P t = %.2f' % (t4 - t3))

        ## Policy evaluation
        V = np.matmul(np.linalg.inv(np.identity(S)-gamma*P), R)
        t5 = time.time()
        print('Policy evaluation, t = %.2f' % (t5 - t4))

        ## Policy improvment

        pi = {}
        for s in States:
            best_V = -10000000
            best_d = 0
            for d in K(s, c, d_actions):
                V_curr = Reward(d, s, c)
                next_w = new_w(s[2], d)
                States_w = All_states_w(next_w)
                tmp = 0
                for s_cand in States_w:
                    tmp += trans_prob(s_cand, s[0][0], d) * V[S_dict[str(s_cand)]]
                V_curr += tmp * gamma
                if V_curr > best_V:
                    best_V = V_curr
                    best_d = d
            pi[str(s)] = best_d

    return pi






def Policy_initializer_a(States, seed = 17):
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
        pi[str(s)] = np.random.choice(5)
    return pi


def possible(next_w, w):
    if w[0] > 0:
        return next_w[0] == w[0]-1
    elif w[1] > 0:
        return next_w[1] == w[1]-1
    elif w[2] > 0:
        return next_w[2] == w[2]-1
    else:
        return next_w[0] == 51 or next_w[1] == 51 or next_w[2] == 51




def Policy_iteration_1(States, c, gamma = 0.1, M = 10, max_iter = 100, seed = 17):

    ## calculate p_D(a|s) by first-order ARA
    S = len(States)
    d_actions = Defender_actions()
    S_dict = {str(s): i for i, s in enumerate(States)}

    new_states_w = All_states_w(np.array([51,0,0])) + All_states_w(np.array([0,51,0])) + All_states_w(np.array([0,0,51]))

    print("Estimating p_D(a|s)")

    pi_A = {}
    for m in tqdm(range(M)):

        pi_a = Policy_initializer_a(States, seed=seed+m)
        prev_pi_a = pi_a

        np.random.seed(seed+m)

        LMH_q0 = dirichlet.rvs([3,3,3]).ravel()
        LMH_q1 = dirichlet.rvs([2.5,2.5,5]).ravel()
        pA1_3 = dirichlet.rvs([2,4,1]).ravel()
        pA4_6_rH_qN = dirichlet.rvs([3,2,1]).ravel()
        pA4_6_rH_qE = dirichlet.rvs([1,2,3]).ravel()
        pA4_6_rL_qN = dirichlet.rvs([3,2]).ravel()
        pA4_6_rL_qE = dirichlet.rvs([1,2]).ravel()
        pA7 = beta.rvs(a=2, b=2)
        pA8 = beta.rvs(a=2, b=2)
        pA9 = beta.rvs(a=2, b=2)

        zeta_r_a0 = expon.rvs(scale=[8, 1, 1])
        zeta_r_a1 = expon.rvs(scale=[1, 1, 8])
        zeta_r_a2 = expon.rvs(scale=[1, 3, 6])
        zeta_r_a3 = expon.rvs(scale=[1, 2, 7])
        zeta_r_a4 = expon.rvs(scale=[1, 4, 5])

        zeta_r_a0 /= zeta_r_a0.sum()
        zeta_r_a1 /= zeta_r_a1.sum()
        zeta_r_a2 /= zeta_r_a2.sum()
        zeta_r_a3 /= zeta_r_a3.sum()
        zeta_r_a4 /= zeta_r_a4.sum()

        zeta_q_a0 = expon.rvs(scale=[8, 1])
        zeta_q_a1 = expon.rvs(scale=[1, 1])
        zeta_q_a2 = expon.rvs(scale=[1, 3])
        zeta_q_a3 = expon.rvs(scale=[1, 2])
        zeta_q_a4 = expon.rvs(scale=[1, 4])

        def theta_given_s_a(theta, q):
            if q == 0:
                return LMH_q0[theta]
            else:
                return LMH_q1[theta]

        def d_given_s(d, s, c):
            q, r, w = s
            prob = 1.
            if np.sum(w) == 0:
                if d[0] == 1:
                    prob *= pA1_3[0]
                elif d[1] == 1:
                    prob *= pA1_3[1]
                else:
                    prob *= pA1_3[2]
            if r[0] > 10:
                if q[0] == 0:
                    if d[3] == 1:
                        prob *= pA4_6_rH_qN[0]
                    elif d[4] == 1:
                        prob *= pA4_6_rH_qN[1]
                    else:
                        prob *= pA4_6_rH_qN[2]
                else:
                    if d[3] == 1:
                        prob *= pA4_6_rH_qE[0]
                    elif d[4] == 1:
                        prob *= pA4_6_rH_qE[1]
                    else:
                        prob *= pA4_6_rH_qE[2]
            else:
                if q[0] == 0:
                    if d[3] == 1:
                        prob *= pA4_6_rL_qN[0]
                    else:
                        prob *= pA4_6_rL_qN[1]
                else:
                    if d[3] == 1:
                        prob *= pA4_6_rL_qE[0]
                    else:
                        prob *= pA4_6_rL_qE[1]

            if r[0] >= np.dot(c, d[3:]):
                if d[6] == 1:
                    prob *= pA7
                if d[7] == 1:
                    prob *= pA8
                if d[8] == 1:
                    prob *= pA9

            return prob

        def attraction_h_a(next_r, a):
            if a == 0:
                if next_r == 9:
                    return zeta_r_a0[0]
                elif next_r == 14:
                    return zeta_r_a0[1]
                else:
                    return zeta_r_a0[2]

            elif a == 1:
                if next_r == 9:
                    return zeta_r_a1[0]
                elif next_r == 14:
                    return zeta_r_a1[1]
                else:
                    return zeta_r_a1[2]

            elif a == 2:
                if next_r == 9:
                    return zeta_r_a2[0]
                elif next_r == 14:
                    return zeta_r_a2[1]
                else:
                    return zeta_r_a2[2]

            elif a == 3:
                if next_r == 9:
                    return zeta_r_a3[0]
                elif next_r == 14:
                    return zeta_r_a3[1]
                else:
                    return zeta_r_a3[2]

            else:
                if next_r == 9:
                    return zeta_r_a4[0]
                elif next_r == 14:
                    return zeta_r_a4[1]
                else:
                    return zeta_r_a4[2]

        def attraction_g_a(next_q, q, d, a):

            if a == 0:
                if next_q == 0:
                    xi_D = zeta_q_a0[0]
                else:
                    xi_D = zeta_q_a0[1]

            elif a == 1:
                if next_q == 0:
                    xi_D = zeta_q_a1[0]
                else:
                    xi_D = zeta_q_a1[1]

            elif a == 2:
                if next_q == 0:
                    xi_D = zeta_q_a2[0]
                else:
                    xi_D = zeta_q_a2[1]

            elif a == 3:
                if next_q == 0:
                    xi_D = zeta_q_a3[0]
                else:
                    xi_D = zeta_q_a3[1]

            else:
                if next_q == 0:
                    xi_D = zeta_q_a4[0]
                else:
                    xi_D = zeta_q_a4[1]

            dqq = 0
            if next_q == 1 and q == 0:
                if d[3] == 1:
                    dqq = 1
                elif np.sum(d[6:]) == 3:
                    dqq = 1
            elif next_q == 0 and q == 1:
                if d[5] == 1:
                    dqq = 1
                elif np.sum(d[6:]) == 0:
                    dqq = 1

            return xi_D + dqq


        def trans_prob_a(next_s, s, a, c):

            next_q, next_r, next_w = next_s

            prob = 0

            for d in Defender_actions():
                next_ww = new_w(s[2], d)
                if np.all(next_s[2] == next_ww):
                    prob_r = attraction_h_a(next_r[0], a)

                    q1 = attraction_g_a(next_q[0], s[0][0], d, a)
                    q2 = attraction_g_a(1 - next_q[0], s[0][0], d, a)
                    prob_q = q1 / (q1 + q2)

                    prob += d_given_s(d, s, c) * prob_r * prob_q

            return prob

        def Reward_a(a, s, c):
            R = 0
            q, r, w = s
            for theta in Random_events():
                for d in Defender_actions():
                    if np.dot(1 - d[6:], w) == 0:
                        R -= theta_given_s_a(theta, s[0][0]) * d_given_s(d, s, c) * \
                             u_D(d, a, theta, c)
            return R



        for iter in tqdm(range(max_iter//2)):

            if iter > 0 and same(States, prev_pi_a, pi_a):
                break
            prev_pi_a = pi_a

            ## Reward array
            R = np.zeros(S)
            for i, s in enumerate(States):
                R[i] = Reward_a(pi_a[str(s)], s, c)

            ## Transition probability matrix
            P = np.zeros((S, S))
            for i, s in enumerate(States):
                a = pi_a[str(s)]
                for j, next_s in enumerate(States):
                    if possible(next_s[2], s[2]):
                        P[i, j] = trans_prob_a(next_s, s, a, c)

            ## Policy evaluation
            V = np.matmul(np.linalg.inv(np.identity(S)-gamma*P), R)

            ## Policy improvment

            pi_a = {}
            for s in States:
                best_V = -10000000
                best_a = 0
                if np.sum(s[2]) > 0:
                    next_w = s[2].copy()
                    next_w[next_w>0] -= 1
                    States_w = All_states_w(next_w)
                else:
                    States_w = new_states_w

                for a in [0,1,2,3,4]:
                    V_curr = Reward_a(a, s, c)
                    tmp = 0
                    for s_cand in States_w:
                        if not possible(s_cand[2], s[2]):
                            print('error')
                        tmp += trans_prob_a(s_cand, s, a, c) * V[S_dict[str(s_cand)]]
                    V_curr += tmp * gamma
                    if V_curr > best_V:
                        best_V = V_curr
                        best_a = a
                pi_a[str(s)] = best_a

        pi_A[m] = pi_a

    pi_as = {}
    for s in States:
        pi_as[str(s)] = [0., 0., 0., 0., 0.]
        for m in range(M):
            pi_as[str(s)][pi_A[m]] += 1.
        pi_as[str(s)] /= M


    def Reward_a_d(d, s, c):
        R = 0
        for theta in Random_events():
            for a in [0,1,2,3,4]:
                R += theta_given_s(theta, s[0][0]) * pi_as[str(s)][a] * \
                     u_D(d, a, theta, c)
        return R



    def trans_prob_a_d(next_s, q, d):

        next_q, next_r, next_w = next_s

        prob = 0

        for a in [0, 1, 2, 3, 4]:
            prob_r = attraction_h(next_r[0], a)

            q1 = attraction_g(next_q[0], q, d, a)
            q2 = attraction_g(1 - next_q[0], q, d, a)
            prob_q = q1 / (q1 + q2)

            prob += pi_as[str(s)][a] * prob_r * prob_q

        return prob


    ## MDP
    print("ARA 1 for defender")

    pi = Policy_initializer(States, c, d_actions, seed=seed)
    prev_pi = pi

    for iter in tqdm(range(max_iter)):

        if iter > 0 and same(States, prev_pi, pi):
            break
        prev_pi = pi

        ## Reward array
        R = np.zeros(S)
        for i, s in enumerate(States):
            R[i] = Reward_a_d(pi[str(s)], s, c)

        ## Transition probability matrix
        P = np.zeros((S, S))
        for i, s in enumerate(States):
            d = pi[str(s)]
            next_w = new_w(s[2], d)
            for j, next_s in enumerate(States):
                if np.all(next_s[2] == next_w):
                    P[i, j] = trans_prob_a_d(next_s, s[0][0], d)

        ## Policy evaluation
        V = np.matmul(np.linalg.inv(np.identity(S) - gamma * P), R)

        ## Policy improvment

        pi = {}
        for s in States:
            best_V = -10000000
            best_d = 0
            for d in K(s, c, d_actions):
                V_curr = Reward_a_d(d, s, c)
                next_w = new_w(s[2], d)
                States_w = All_states_w(next_w)
                tmp = 0
                for s_cand in States_w:
                    tmp += trans_prob_a_d(s_cand, s[0][0], d) * V[S_dict[str(s_cand)]]
                V_curr += tmp * gamma
                if V_curr > best_V:
                    best_V = V_curr
                    best_d = d
            pi[str(s)] = best_d

    return pi


def main():
    States = All_states()

    c  = np.array([4., 8., 12., 2., 2., 2.])

    MDP = Policy_iteration(States, c, gamma = 0.1, max_iter = 100, seed = 17)

    return MDP


if __name__ == '__main__':
    #print(Defender_actions())
    #print(len(All_states()))
    #print(Defender_actions())
    pi = main()
    cnt = 0
    for s in All_states():
        print(s, pi[str(s)])
        cnt += 1
        if cnt > 10:
            break
