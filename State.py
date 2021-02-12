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


def Random_events(s = None):
    """
    Possible random events at state s
    Args:
        s = [q, r, w]: Current State
    Returns:
        thetas: a list of random events
    """
    return Categorical_vectors(3)


def Op_conditions(s = None):
    """
    Possible operational conditions of next state at current state s
    Args:
        s = [q, r, w]: Current State
    Returns:
        qs: a list of operational conditions
    """
    return Categorical_vectors(2)


def Resources(s = None):
    """
    All possible resource states.
    Args:
        s = [q, r, w]: Current State
    Returns:
        qs: a list of resource states.
    """

    return Categorical_vectors(1, [9., 14., 19.])



def Attacker_actions(s = None):
    """
        Possible attacker's actions at state s
        Args:
            s = [q, r, w]: Current State
        Returns:
            actions: a list of attacker's actions
        """
    return [0,1,2,3,4]



def Defender_actions(s = None):
    """
    Possible defender's actions at state s
    Args:
        s = [q, r, w]: Current State
    Returns:
        actions: a list of defender's actions
    """
    actions = []
    d1_3 = [np.array([1., 0., 0.]),
            np.array([0., 1., 0.]),
            np.array([0., 0., 1.])]
    d4_6 = [np.array([1., 0., 0.]),
            np.array([0., 1., 0.]),
            np.array([0., 0., 1.])]
    d7_9 = [np.array([0, 0, 1]),
             np.array([0, 1, 0]),
             np.array([0, 1, 1]),
             np.array([1, 0, 0]),
             np.array([1, 0, 1]),
             np.array([1, 1, 0]),
             np.array([1, 1, 1])]

    ITER = itertools.product(d1_3, d4_6, d7_9)
    for i1, i2, i3 in ITER:
        actions.append(np.concatenate([i1, i2, i3]))
    return actions




def K(s, c, actions=Defender_actions()):
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

    # np.all((np.matmul(c,d)<=r))
    for d in actions:
        if  np.dot(c, d[3:]) <= r[0] and np.dot(w, 1-d[:3])==0:
            actions_satisfied.append(d)
    return actions_satisfied
