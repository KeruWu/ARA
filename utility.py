import numpy as np
from ARA import *
from transition import *
from State import *

def u_D(d, a, theta, c):
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
    u = 0

    ## attack effect

    if theta == 0:
        k_a = 0.8
    elif theta == 1:
        k_a = 1
    else:
        k_a = 1.2

    p = [0., 0., 0., 1., 1.]

    ## number of TSA agents
    if d[0] == 1:
        n_TSA = 6
        p[1] = 0.01
        p[2] = 0.1
    elif d[1] == 1:
        n_TSA = 12
        p[1] = 0.05
        p[2] = 0.01
    else:
        n_TSA = 18
        p[1] = 0.1
        p[2] = 0.01

    ## number of non-TSA agents
    if d[3] == 1:
        n_nonTSA = 4
    elif d[4] == 1:
        n_nonTSA = 8
    else:
        n_nonTSA = 12

    n_staff = n_TSA + n_nonTSA


    cost_a = [0., 100000., 10000., 0., 0.]

    n_camera = np.sum(d[6:])

    if n_staff <= 10:
        cost_a[3] = 5000
        cost_a[4] = 5000
        cost_passenger = 0.1 * (10-n_camera)
    elif n_staff <= 20:
        cost_a[3] = 2100
        cost_a[4] = 2100
        cost_passenger = 0.07 * (7-n_camera)
    else:
        cost_a[3] = 1000
        cost_a[4] = 1000
        cost_passenger = 0.05 * (5-n_camera)


    u -= cost_a[a] * p[a] * k_a
    u -= cost_passenger
    u -= np.dot(c, d[3:])

    return u


def Reward(d, s, c, order=0):
    """
    Defender's immediate reward of action d at state s.
    Args:
        d: Defender's actions
        s = [q, r, w]: State
        c (nr * nd): cost of defender's each action: (c1: new, c2: annual)
        order: Order of ARA. Currently only 0 and 1 are available.
    Returns:
        r: Reward value.
    """
    R = 0
    for theta in Random_events():
        for a in [0,1,2,3,4]:
            R += theta_given_s(theta, s[0][0]) * a_given_s(a, s[0][0]) *\
                u_D(d, a, theta, c)
    return R




