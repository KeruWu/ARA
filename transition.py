import numpy as np
from ARA import *
from State import *

def theta_given_s(theta, q):
    """
    Probability of an random event theta given current state s.
    Args:
        theta: Random event
        s = [q, r, w]: State
    Returns:
        Unnormalized probability of the random event.
    """
    if q == 0:
        return .3333
    else:
        if theta == 0:
            return 0.25
        elif theta == 1:
            return 0.25
        else:
            return 0.5


def new_w(w, d):
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

    if w.sum() > 0:
        next_w = w.copy()
        next_w[next_w > 0] -= 1
        return next_w
    else:
        if d[0] == 1:
            return np.array([51,0,0])
        elif d[1] == 1:
            return np.array([0,51,0])
        else:
            return np.array([0,0,51])



def attraction_h(next_r, a):
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
    if a == 0:
        if next_r == 9:
            return 0.8
        elif next_r == 14:
            return 0.1
        else:
            return 0.1

    elif a == 1:
        if next_r == 9:
            return 0.1
        elif next_r == 14:
            return 0.1
        else:
            return 0.8

    elif a == 2:
        if next_r == 9:
            return 0.1
        elif next_r == 14:
            return 0.3
        else:
            return 0.6

    elif a == 3:
        if next_r == 9:
            return 0.1
        elif next_r == 14:
            return 0.2
        else:
            return 0.7

    else:
        if next_r == 9:
            return 0.1
        elif next_r == 14:
            return 0.4
        else:
            return 0.5


def attraction_g(next_q, q, d, a):
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

    if a == 0:
        if next_q == 0:
            xi_D = 8
        else:
            xi_D = 1

    elif a == 1:
        xi_D = 1

    elif a == 2:
        if next_q == 0:
            xi_D = 1
        else:
            xi_D = 3

    elif a == 3:
        if next_q == 0:
            xi_D = 1
        else:
            xi_D = 2

    else:
        if next_q == 0:
            xi_D = 1
        else:
            xi_D = 4

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



def trans_prob(next_s, q, d):
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

    A_actions = [0, 1, 2, 3, 4]

    prob = 0

    for a in A_actions:

        prob_r = attraction_h(next_r[0], a)

        q1 = attraction_g(next_q[0], q, d, a)
        q2 = attraction_g(1-next_q[0], q, d, a)
        prob_q = q1 / (q1 + q2)

        prob += a_given_s(a, q) * prob_r * prob_q

    return prob

