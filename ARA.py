import numpy as np

def a_given_s(a, s, order = 0):
    """
    Defender's belief about the attacker's actions.
    Args:
        a: Attacker's actions
        s = [q, r, w]: State
        order: Order of ARA. Currently only 0 and 1 are available.
    Returns:
        Probability of attacker's action given current state.
    """
    if order == 0:
        ## Zeroth-Order ARA
        return 1
    elif order == 1:
        ## First-Order ARA
        return 1
    else:
        raise ValueError('Invalid ARA order')
