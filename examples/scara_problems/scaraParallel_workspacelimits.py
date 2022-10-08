"""
This function calculates the workspace limitations of the scara robot, such
that the workspace limits of the robot can be plotted.
"""

import numpy as np
from matplotlib import pyplot as plt


def workspace_boundary(l_p=0.18, l_d=0.28, d=0.15):
    """
    Calculates the workspace boundary of the scara robot.

    Input:
    d:      distance between the two joints
    l_p:    length of one of the bars
    l_d:    length of the other bar

    Output:
    X:      X-coordinates of workspace boundary
    Y:      Y-coordinates of workspace boundary
    """
    X = []
    Y = []

    t = np.linspace(48*np.pi/180, 117*np.pi/180, 200)
    s1 = np.cos(t)*l_d-d/2-l_p*np.cos(30/180*np.pi)
    s2 = np.sin(t)*l_d-l_p*np.sin(30/180*np.pi)
    X = np.concatenate((X, s1))
    Y = np.concatenate((Y, s2))

    t = np.linspace(20.5*np.pi/180, np.arccos(d/2/(l_p+l_d)), 200)
    s1 = -np.cos(t)*(l_p+l_d)+d/2
    s2 = np.sin(t)*(l_p+l_d)
    X = np.concatenate((X, s1))
    Y = np.concatenate((Y, s2))

    s1 = np.cos(t)*(l_p+l_d)-d/2
    X = np.concatenate((X, np.flip(s1)))
    Y = np.concatenate((Y, np.flip(s2)))

    t = np.linspace(39*np.pi/180, 117*np.pi/180, 200)
    s1 = -np.cos(t)*l_d+d/2+l_p*np.cos(30/180*np.pi)
    s2 = np.sin(t)*l_d-l_p*np.sin(30/180*np.pi)
    X = np.concatenate((X, np.flip(s1)))
    Y = np.concatenate((Y, np.flip(s2)))

    t = np.linspace(44*np.pi/180, 75*np.pi/180, 200)
    zeta = 18.05*np.pi/180
    z = np.sqrt(l_p**2 * np.sin(zeta)**2 + (l_d-l_p*np.cos(zeta))**2)
    s1 = np.cos(t)*z-d/2
    s2 = np.sin(t)*z
    X = np.concatenate((X, s1))
    Y = np.concatenate((Y, s2))

    return X, Y
