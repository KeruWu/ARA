import numpy as np

"""

This file includes functions which return a list of possible 
defender's actions, attacker's actions, random events and 
resource array at state s.

"""

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
    tmp = np.zeros(c.shape[1])
    for i in range(len(tmp), len(tmp)-len(w), -1):
        if w[i-(len(tmp)-len(w))-1]>0:
            tmp[i] = 1
    for next_r in resources:
        if np.all(next_r-np.matmul(c,tmp)>=0):
            resources_satisfied.append(next_r)
    return resources_satisfied


def Attacker_actions(s):
    """
        Possible attacker's actions at state s
        Args:
            s = [q, r, w]: Current State
        Returns:
            actions: a list of attacker's actions
        """
    actions = []
    return actions


def Defender_actions(s):
    """
    Possible defender's actions at state s
    Args:
        s = [q, r, w]: Current State
    Returns:
        actions: a list of defender's actions
    """
    actions = []
    return actions


def K(s, c):
    """
    Possible defender's actions under knapsack constraints
    Args:
        s = [q, r, w]: Current State
        c (nr * nd): cost of defender's each action
    Returns:
        actions_satisfied: a list of defender's actions
    """
    actions = Defender_actions(s)
    actions_satisfied = []
    q, r, w = s
    for d in actions:
        satisfy = True
        satisfy = satisfy and np.all((np.matmul(c,d)-r)<=0)
        satisfy = satisfy and np.all((np.dot(w, d[len(d)-len(w):]))==0)
        if satisfy:
            actions_satisfied.append(d)
    return actions_satisfied

