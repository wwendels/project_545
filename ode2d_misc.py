# miscellaneous functions for analysis 2D ODE systems
# Mike Wendels, last modified 2/19/2023

import numpy as np
import matplotlib.pyplot as plt

def diagramVI(modelclass,xmesh,label):

    xmin, xmax, _, nx, xlab = xmesh 
    x = np.linspace(xmin,xmax,nx,endpoint=True)

    plt.figure()
    plt.plot(x, -modelclass.f(np.array([x,modelclass.ninf(x)]))[0]/modelclass.C)
    plt.xlim([xmin, xmax])
    plt.ylim([-200,200])
    plt.xlabel(xlab)
    plt.ylabel("$I$")
    plt.savefig("VI-diagram_"+label+".png")

def inList(x,X,tol=1e-2):

    n = len(X)
    flag = False
    for i in range(n):
        if np.linalg.norm(np.array(X[i])-np.array(x),2) < tol:
            flag = True

    return flag