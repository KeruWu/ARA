import numpy as np
import itertools

"""

This file includes functions which return a list of possible 
defender's actions, attacker's actions, random events and 
resource array at state s.

"""

def Binary_vectors(length):
    """
    Generate all binary vectors of specified length
    Args:
        length: Length of the binary vector
    Returns:
        l: A list of binary vectors
    """
    l = []
    B = itertools.product([0,1], repeat=length)
    for b in B:
        l.append(np.array(b))
    return l


def One_hot_vectors(length):
    """
    Generate all one-hot vectors of specified length
    Args:
        length: Length of the binary vector
    Returns:
        l: A list of binary vectors
    """
    l = []
    for i in range(length):
        tmp = np.zeros(length)
        tmp[i] = 1
        l.append(tmp)
    return l


def Categorical_vectors(num, levels = None):
    """
    Generate all categories for a categorical variable
    Args:
        num: Number of categories
        levels: Numerical value for each category
    Returns:
        l: A list of vectors of length 1
    """
    l = []
    if levels is None:
        levels = list(np.arange(num))
    for i in levels:
        tmp = np.array([i])
        l.append(tmp)
    return l


def Random_events(s = None, Case_Study = 'Airport Security'):
    """
    Possible random events at state s
    Args:
        s = [q, r, w]: Current State
    Returns:
        thetas: a list of random events
    """
    if Case_Study == 'Airport Security':
        return Categorical_vectors(3)
    else:
        thetas = []
        return thetas


def Op_conditions(s = None, Case_Study = 'Airport Security'):
    """
    Possible operational conditions of next state at current state s
    Args:
        s = [q, r, w]: Current State
    Returns:
        qs: a list of operational conditions
    """

    if Case_Study == 'Airport Security':
        return Categorical_vectors(5)
    else:
        qs = []
        return qs


def Resources(s = None, Case_Study = 'Airport Security'):
    """
    All possible resource states.
    Args:
        s = [q, r, w]: Current State
    Returns:
        qs: a list of resource states.
    """

    if Case_Study == 'Airport Security':
        rs = []
        r1 = Categorical_vectors(1, [1000, 2000, 3000])
        r2 = Categorical_vectors(1, [50000, 75000, 100000])
        #r3 = Categorical_vectors(1, [1e6, 2e6, 3e6])
        ITER = itertools.product(r1, r2)#, r3)
        for i1, i2, i3 in ITER:
            rs.append(np.concatenate([i1, i2]))#, i3]))
        return rs
    else:
        rs = []
        return rs


def R0(s, c, resources):
    """
    Possible resource arrays of next state at current state s
    Args:
        s = [q, r, w]: Current State
        c (nr * nd): cost of defender's each action
        resources: All possible resource states.
    Returns:
        resources_satisfied: a list of resource arrays
    """
    resources_satisfied = []
    q, r, w = s
    c1, c2 = c
    curr_c = c1.copy()
    for i in range(len(w)):
        if w[i] == 0:
            curr_c[:, i+len(c1.shape[1])-len(w)] = c1[:, i+len(c1.shape[1])-len(w)]
        else:
            curr_c[:, i+len(c1.shape[1])-len(w)] = c2[:, i+len(c1.shape[1])-len(w)]
    tmp = np.zeros(curr_c.shape[1])
    for i in range(len(tmp), len(tmp)-len(w), -1):
        if w[i-(len(tmp)-len(w))-1]>0:
            tmp[i] = 1
    for next_r in resources:
        if np.all(next_r-np.matmul(curr_c,tmp)>=0):
            resources_satisfied.append(next_r)
    return resources_satisfied


def Attacker_actions(s = None, Case_Study = 'Airport Security'):
    """
        Possible attacker's actions at state s
        Args:
            s = [q, r, w]: Current State
        Returns:
            actions: a list of attacker's actions
        """
    actions = []
    if Case_Study == 'Airport Security':
        return Binary_vectors(3)
    return actions


def Defender_actions(s = None, Case_Study = 'Airport Security'):
    """
    Possible defender's actions at state s
    Args:
        s = [q, r, w]: Current State
    Returns:
        actions: a list of defender's actions
    """
    if Case_Study == 'Airport Security':
        actions = []
        d1 = Binary_vectors(1)
        d2_5 = One_hot_vectors(4)
        d6_8 = Binary_vectors(3)
        d9_10 = One_hot_vectors(2)
        d11 = Binary_vectors(1)
        d12 = Binary_vectors(1)

        ITER = itertools.product(d1, d2_5, d6_8, d9_10, d11, d12)
        for i1, i2, i3, i4, i5, i6 in ITER:
            actions.append(np.concatenate([i1, i2, i3, i4, i5, i6]))
        return actions
    else:
        actions = []
        return actions



def K(s, c, actions):
    """
    Possible defender's actions under knapsack constraints
    Args:
        s = [q, r, w]: Current State
        c (nr * nd): cost of defender's each action: (c1: new, c2: annual)
        actions: List of Defender actions
    Returns:
        actions_satisfied: a list of defender's actions
    """
    actions_satisfied = []
    q, r, w = s
    c1, c2 = c
    curr_c = c1.copy()
    for i in range(len(w)):
        if w[i] == 0:
            curr_c[:, i+len(c1.shape[1])-len(w)] = c1[:, i+len(c1.shape[1])-len(w)]
        else:
            curr_c[:, i+len(c1.shape[1])-len(w)] = c2[:, i+len(c1.shape[1])-len(w)]

    for d in actions:
        satisfy = True
        satisfy = satisfy and np.all((np.matmul(curr_c,d)-r)<=0)
        satisfy = satisfy and np.all((np.dot(w, 1-d[len(d)-len(w):]))==0)
        if satisfy:
            actions_satisfied.append(d)
    return actions_satisfied
