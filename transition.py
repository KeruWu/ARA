import numpy as np
from ARA import a_given_s
from State import Attacker_actions, Op_conditions, Resources

def theta_given_s(theta, s):
    """
    Probability of an random event theta given current state s.
    Args:
        theta: Random event
        s = [q, r, w]: State
    Returns:
        Unnormalized probability of the random event.
    """
    return 1


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


def attraction_h(next_r, next_w, d, a, s):
    """
    Attraction function of resource (h in the paper).
    Args:
        next_r: Probable resource array in the next epoch.
        next_w: Multi-period commitments in the next epoch.
        d: Defender's actions
        a: Attacker's actions
        s = [q, r, w]: Current State
    Returns:
        Attraction value.
    """
    return 1


def attraction_q(next_q, next_r, next_w, d, a, s):
    """
    Attraction function of operational conditions (h in the paper).
    Args:
        next_q: Operational conditions in the next epoch.
        next_r: Probable resource array in the next epoch.
        next_w: Multi-period commitments in the next epoch.
        d: Defender's actions
        a: Attacker's actions
        s = [q, r, w]: Current State
    Returns:
        Attraction value.
    """
    return 1


def trans_prob(next_s, d, s, m, tau, c, order = 0):
    """
    Probability of decision d from state s to state next_s
    Args:
        next_s = [next_q, next_r, next_w]: Next State
        d: Defender's actions
        s = [q, r, w]: Current State
        m: Number of non multi-period commitments. (i.e. The first m defender's actions are not multi-period)
        tau: An array denoting the length of each multi-period commitment.
        c (nr * nd): cost of defender's each action
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
        for r_cand in Resources(s, c):
            r_normalize += attraction_h(r_cand, next_w, d, a, s)
        prob_r = attraction_h(next_r, next_w, d, a, s) / r_normalize
        q_normalize = 0
        for q_cand in Op_conditions(s):
            q_normalize += attraction_q(q_cand, next_r, next_w, d, a, s)
        prob_q = attraction_q(next_q, next_r, next_w, d, a, s) / q_normalize
        prob += a_given_s(a, s, order) * prob_r * prob_q

    return prob