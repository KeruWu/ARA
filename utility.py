import numpy as np
from ARA import a_given_s
from transition import theta_given_s
from State import Random_events, Attacker_actions


def u_D(d, a, s, theta):
    """
    Utility function of the defender.
    Args:
        d: Defender's actions
        a: Attacker's actions
        s = [q, r, w]: State
        theta: Random event
    Returns:
        Utility value. Yes
    """
    q, r, w = s
    ## Attacker Favorable
    if theta[0] == 0:
        u = 1
    ## Neutral
    elif theta[0] == 1:
        u = 1
    ## Defender Favorable
    else:
        u = 1
    return u


def Reward(d, s, order=0):
    """
    Defender's immediate reward of action d at state s.
    Args:
        d: Defender's actions
        s = [q, r, w]: State
        order: Order of ARA. Currently only 0 and 1 are available.
    Returns:
        r: Reward value.
    """
    r = 0
    for theta in Random_events(s):
        for a in Attacker_actions(s):
            r += theta_given_s(theta, s) * a_given_s(a, s, order=order) *\
                u_D(d, a, s, theta)
    return r



