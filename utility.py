import numpy as np
from ARA import a_given_s
from transition import theta_given_s
from State import Random_events, Attacker_actions


def u_D(d, a, s, theta, c):
    """
    Utility function of the defender.
    Args:
        d: Defender's actions
        a: Attacker's actions
        s = [q, r, w]: State
        theta: Random event
        c (nr * nd): cost of defender's each action: (c1: new, c2: annual)
    Returns:
        Utility value. Yes
    """
    q, r, w = s

    ## current action cost
    c1, c2 = c
    curr_c = c1.copy()
    for i in range(len(w)):
        if w[i] == 0:
            curr_c[:, i+len(d)-len(w)] = c1[:, i+len(d)-len(w)]
        else:
            curr_c[:, i+len(d)-len(w)] = c2[:, i+len(d)-len(w)]

    ## Attacker Favorable
    if theta[0] == 0:
        k_d = 0.5
        k_a = 2
    ## Neutral
    elif theta[0] == 1:
        k_d = 1
        k_a = 1
    ## Defender Favorable
    else:
        k_d = 2
        k_a = 0.5


    ## number of TSA agents
    n_TSA = 0
    if d[1] == 1:
        n_TSA = 5
    elif d[2] == 1:
        n_TSA = 10
    elif d[3] == 1:
        n_TSA = 15
    elif d[4] == 1:
        n_TSA = 20

    ## number of non-TSA agents
    n_nonTSA = 0
    if d[8] == 1:
        n_nonTSA = 5
    elif d[9] == 1:
        n_nonTSA = 10

    n_staff = n_TSA + n_nonTSA

    ca1 = 100
    if n_staff <= 10:
        ca2 = 10
        ca3 = 0.5
    elif n_staff <= 20:
        ca2 = 7
        ca3 = 0.3
    else:
        ca2 = 5
        ca3 = 0.2

    return k_d * ((r[0]+r[1])/1000 - np.dot(curr_c.sum(0), d)) - k_a * (a[0]*ca1 + a[1]*ca2 + a[2]*ca3)


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



