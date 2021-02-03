import numpy as np

def a_given_s(a, s, theta=None, order = 0, Case_Study = 'Airport Security'):
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
    if order == 0:
        ## Zeroth-Order ARA
        if Case_Study == 'Airport Security':
            prob = 1
            if a[0] == 1:
                prob *= 1e-7
            if a[1] == 1:
                prob *= 5e-6
            if a[2] == 1:
                prob *= 0.5
            return prob
        else:
            return 1

    elif order == 1:
        ## First-Order ARA
        return 1

    else:
        raise ValueError('Invalid ARA order')
