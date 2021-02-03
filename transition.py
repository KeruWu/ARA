import numpy as np
from ARA import a_given_s
from State import Attacker_actions, Op_conditions, R0

def theta_given_s(theta, s):
    """
    Probability of an random event theta given current state s.
    Args:
        theta: Random event
        s = [q, r, w]: State
    Returns:
        Unnormalized probability of the random event.
    """
    q, r, w = s
    return .3333


def new_w(d, m, s, tau):
    """
    Multi-period commitments in the next epoch.
    Args:
        d: Defender's actions
        m: Number of non multi-period commitments. (i.e. The first m defender's actions are not multi-period)
        s = [q, r, w]: Current State
        tau: An array denoting the length of each multi-period commitment.
    Returns:
        next_w: Number of decision epochs remaining in the next epoch.
    """
    w = s[2]
    next_w = np.zeros_like(w)
    for i in range(m, len(d)):
        if w[i-m] > 0:
            next_w[i-m] = w[i-m]
        elif d[i] > 0:
            next_w[i-m] = tau[i-m]
    return next_w


def attraction_h(next_r, next_w, d, a, s, rho_da, rho_dq, h_above, h_below, dict_r, thres=5):
    """
    Attraction function of resource (h in the paper).
    Args:
        next_r: Probable resource array in the next epoch.
        next_w: Multi-period commitments in the next epoch.
        d: Defender's actions
        a: Attacker's actions
        s = [q, r, w]: Current State
        rho_da: A map mapping from (d_i, a_j) to response quality
        rho_dq: A map mapping from (d_i, q) to response quality
        h_above: attraction value when response quality is above threshold
        h_below: attraction value when response quality is below threshold
        dict_r: map resource to corresponding level.
        thres: Threshold for a good response.
    Returns:
        Attraction value.
    """
    q, r, w = s
    rho_1 = np.dot(np.matmul(d, rho_da), a)
    rho_2 = np.sum(rho_dq[:,q[0]])

    rho = rho_1 + rho_2
    h = 1
    if rho > thres:
        for i in range(len(r)):
            h *= h_above[dict_r[i][r[i]]][dict_r[i][next_r[i]]]
        return h
    else:
        for i in range(len(r)):
            h *= h_below[dict_r[i][r[i]]][dict_r[i][next_r[i]]]
        return h


def attraction_g(next_q, next_r, next_w, d, a, s, rho_da, rho_dq, g_above, g_below, thres=5):
    """
    Attraction function of operational conditions (g in the paper).
    Args:
        next_q: Operational conditions in the next epoch.
        next_r: Probable resource array in the next epoch.
        next_w: Multi-period commitments in the next epoch.
        d: Defender's actions
        a: Attacker's actions
        s = [q, r, w]: Current State
        rho_da: A map mapping from (d_i, a_j) to response quality
        rho_dq: A map mapping from (d_i, q) to response quality
        g_above: attraction value when response quality is above threshold
        g_below: attraction value when response quality is below threshold
        thres: Threshold for a good response.
    Returns:
        Attraction value.
    """
    q, r, w = s
    rho_1 = np.dot(np.matmul(d, rho_da), a)
    rho_2 = np.sum(rho_dq[:, q[0]])

    rho = rho_1 + rho_2
    if rho > thres:
        g = g_above[q[0]][next_q[0]]
        return g
    else:
        g = g_below[q[0]][next_q[0]]
        return g


"""
def h_normalize(next_w, d, s, c):
    
    Precompute denominator values for attraction function h
    Args:
        next_w = Multi-period commitments in the next epoch.
        d: Defender's actions
        s = [q, r, w]: Current State
        c (nr * nd): cost of defender's each action
    Returns:
        all_h_normalize: A map which maps string representation of action to its corresponding value.
    
    A_actions = Attacker_actions(s)
    all_h_normalize = {}
    for a in A_actions:
        r_normalize = 0
        for r_cand in R0(s, c):
            r_normalize += attraction_h(r_cand, next_w, d, a, s)
        all_h_normalize[str(a)] = r_normalize
    return all_h_normalize



def g_normalize(next_r, next_w, d, s):
    
     Precompute denominator values for attraction function g
     Args:
         next_r: Probable resource array in the next epoch.
         next_w = Multi-period commitments in the next epoch.
         d: Defender's actions
         s = [q, r, w]: Current State
     Returns:
         all_g_normalize: A map which maps string representation of action to its corresponding value.
    
    A_actions = Attacker_actions(s)
    all_g_normalize = {}
    for a in A_actions:
        q_normalize = 0
        for q_cand in Op_conditions(s):
            q_normalize += attraction_g(q_cand, next_r, next_w, d, a, s)
        all_g_normalize[str(a)] = q_normalize
    return all_g_normalize
"""


def trans_prob(next_s, d, s, m, tau, c, rho_da, rho_dq, h_above, h_below, g_above, g_below, dict_r, order = 0):
    """
    Probability of decision d from state s to state next_s
    Args:
        next_s = [next_q, next_r, next_w]: Next State
        d: Defender's actions
        s = [q, r, w]: Current State
        m: Number of non multi-period commitments. (i.e. The first m defender's actions are not multi-period)
        tau: An array denoting the length of each multi-period commitment.
        c (nr * nd): cost of defender's each action
        h_above: attraction value when response quality is above threshold
        h_below: attraction value when response quality is below threshold
        g_above: attraction value when response quality is above threshold
        g_below: attraction value when response quality is below threshold
        dict_r: map resource to corresponding level.
        order: Order of ARA. Currently only 0 and 1 are available.
    Returns:
        prob: Probability.
    """

    next_q, next_r, next_w = next_s
    q, r, w = s

    if not np.all(next_w==new_w(d, m, s, tau)):
        return 0

    A_actions = Attacker_actions(s)
    prob = 0

    for a in A_actions:
        r_normalize = 0
        #for r_cand in R0(s, c):
        #    r_normalize += attraction_h(r_cand, next_w, d, a, s, rho_da, rho_dq, h_above, h_below, dict_r)
        prob_r = attraction_h(next_r, next_w, d, a, s, rho_da, rho_dq, h_above, h_below, dict_r) #/ r_normalize
        #prob_r = attraction_h(next_r, next_w, d, a, s) / dict_h(str(a))

        q_normalize = 0
        #for q_cand in Op_conditions(s):
        #    q_normalize += attraction_g(q_cand, next_r, next_w, d, a, s, rho_da, rho_dq, g_above, g_below)
        prob_q = attraction_g(next_q, next_r, next_w, d, a, s, rho_da, rho_dq, g_above, g_below) #/ q_normalize

        prob += a_given_s(a, s, order) * prob_r * prob_q

    return prob