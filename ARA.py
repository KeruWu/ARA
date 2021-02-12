import numpy as np



def a_given_s(a, q):
    """
    Defender's belief about the attacker's actions.
    Args:
        a: Attacker's actions
        s = [q, r, w]: State
        theta: State of nature
        order: Order of ARA. Currently only 0 and 1 are available.
    Returns:
        Probability of attacker's action given current state.
    """
    if q == 0:
        if a == 0:
            return 1.-1e-6
        elif a == 1:
            return 1e-7
        elif a == 2:
            return 2e-7
        elif a == 3:
            return 3e-7
        else:
            return 4e-7
    elif q == 1:
        if a == 0:
            return 1-1e-5
        elif a == 1:
            return 1e-6
        elif a == 2:
            return 2e-6
        elif a == 3:
            return 3e-6
        else:
            return 4e-6
