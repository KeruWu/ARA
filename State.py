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


def Random_events(s):
    """
    Possible random events at state s
    Args:
        s = [q, r, w]: Current State
    Returns:
        thetas: a list of random events
    """
    thetas = []
    return thetas

def Op_conditions(s):
    """
    Possible operational conditions of next state at current state s
    Args:
        s = [q, r, w]: Current State
    Returns:
        qs: a list of random events
    """
    qs = []
    return qs

def Resources(s, c):
    """
    Possible resource arrays of next state at current state s
    Args:
        s = [q, r, w]: Current State
        c (nr * nd): cost of defender's each action
    Returns:
        resources: a list of resource arrays
    """
    resources = []
    resources_satisfied = []
    q, r, w = s
    c1, c2 = c
    curr_c = np.zeros_like(c1)
    for i in range(len(w)):
        if w[i] == 0:
            curr_c[:, i] = c1[:, i]
        else:
            curr_c[:, i] = c2[:, i]
    tmp = np.zeros(curr_c.shape[1])
    for i in range(len(tmp), len(tmp)-len(w), -1):
        if w[i-(len(tmp)-len(w))-1]>0:
            tmp[i] = 1
    for next_r in resources:
        if np.all(next_r-np.matmul(curr_c,tmp)>=0):
            resources_satisfied.append(next_r)
    return resources_satisfied


def Attacker_actions(s, Case_Study = 'Airport Security'):
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


def Defender_actions(s, Case_Study = 'Airport Security'):
    """
    Possible defender's actions at state s
    Args:
        s = [q, r, w]: Current State
    Returns:
        actions: a list of defender's actions
    """
    actions = []
    if Case_Study == 'Airport Security':
        return actions
    return actions


def K(s, c):
    """
    Possible defender's actions under knapsack constraints
    Args:
        s = [q, r, w]: Current State
        c (nr * nd): cost of defender's each action: (c1: new, c2: annual)
    Returns:
        actions_satisfied: a list of defender's actions
    """
    actions = Defender_actions(s)
    actions_satisfied = []
    q, r, w = s
    c1, c2 = c
    curr_c = np.zeros_like(c1)
    for i in range(len(w)):
        if w[i] == 0:
            curr_c[:, i] = c1[:, i]
        else:
            curr_c[:, i] = c2[:, i]

    for d in actions:
        satisfy = True
        satisfy = satisfy and np.all((np.matmul(curr_c,d)-r)<=0)
        satisfy = satisfy and np.all((np.dot(w, 1-d[len(d)-len(w):]))==0)
        if satisfy:
            actions_satisfied.append(d)
    return actions_satisfied

