# functions for phase plane and time series analysis for analysis 2D ODE systems
# Mike Wendels, last modified 3/17/2023

import numpy as np
from scipy.optimize import fsolve
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.optimize import fmin


def inList(x,X,tol=1e-2):
    # checks whether element x is already present in array X up to tolerance tol
    # function used in searching for equilibria

    n = len(X)
    flag = False
    for i in range(n):
        if np.linalg.norm(np.array(X[i])-np.array(x),2) < tol:
            flag = True

    return flag

def phasePlane(modelclass,xmesh,ymesh,t=0):
    # extracts Xdot for the given points X

    xmin, xmax, nx = xmesh 
    ymin, ymax, ny = ymesh
    # dx, dy = (xmax-xmin)/float(nx-1), (ymax-ymin)/float(ny-1)
    x = np.linspace(xmin,xmax,nx,endpoint=True)
    y = np.linspace(ymin,ymax,ny,endpoint=True)

    X, Y = np.meshgrid(x,y)
    u, v = np.zeros_like(X), np.zeros_like(X)
    for i in range(nx):
        for j in range(ny):
            u[i,j],v[i,j] = modelclass.f(np.array([X[i,j],Y[i,j]]),t)

    return u,v,X,Y


def nullclines(modelclass,xmesh,ymesh,t=0,method=1):
    # searches for the nullclines, extracted either on the given x-mesh 
    # (method == 1) or the given y-mesh (method == 2)

    xmin, xmax, nx = xmesh 
    ymin, ymax, ny = ymesh
    x = np.linspace(xmin,xmax,nx,endpoint=True)
    y = np.linspace(ymin,ymax,ny,endpoint=True)

    # nx,ny = len(x),len(y)
    NC1 = []
    NC2 = []
    
    if method == 1: # search for y coordinate for each given x coordinate
        for i in range(nx):
            xi = x[i]
            f1 = lambda yy : modelclass.f1(xi,yy,t)
            f2 = lambda yy : modelclass.f2(xi,yy,t)
            yi_nc1 = fsolve(f1,x0=0)
            yi_nc2 = fsolve(f2,x0=0)
            NC1.append(np.array([xi,yi_nc1[0]]))
            NC2.append(np.array([xi,yi_nc2[0]]))

    elif method == 2: # search for x coordinate for each given y coordinate
        for j in range(ny):
            yj = y[j]
            f1 = lambda xx : modelclass.f1(xx,yj,t)
            f2 = lambda xx : modelclass.f2(xx,yj,t)
            xj_nc1 = fsolve(f1,x0=0)
            xj_nc2 = fsolve(f2,x0=0)
            NC1.append(np.array([xj_nc1,yj[0]]))
            NC2.append(np.array([xj_nc2,yj[0]]))

    NC1 = np.array(NC1)
    NC2 = np.array(NC2)

    fnc1 = InterpolatedUnivariateSpline(NC1[:,0], NC1[:,1])
    fnc2 = InterpolatedUnivariateSpline(NC2[:,0], NC2[:,1])

    return fnc1, fnc2, x, y


def equilibria(modelclass,xmesh,ymesh,t=0,tol=1e-3,sigfig=3):
    # searches for equilibria by using the given mesh points as the initial conditions for fsolve

    xmin, xmax, nx = xmesh 
    ymin, ymax, ny = ymesh
    x = np.linspace(xmin,xmax,nx,endpoint=True)
    y = np.linspace(ymin,ymax,ny,endpoint=True)

    f = lambda xy : modelclass.f(xy,t)

    X, Y = np.meshgrid(x,y)
    E = []
    ni, nj = X.shape
    for i in range(ni):
        for j in range(nj):
            Eij = fsolve(f,x0=np.array([X[i,j],Y[i,j]]))
            Eij = Eij.round(sigfig)
            flag = inList(Eij,E)
            if not flag:
                if np.linalg.norm(modelclass.f(np.array([Eij[0],Eij[1]]),t),2) <= tol:
                    E.append(Eij)

    return np.array(E)


def equilibria2(modelclass,xmesh,ymesh,t=0,x_ref=np.array([]),tol=1e-3,sigfig=3):
    # find equilibria by applying fsolve to find a difference of 0 between the nullclines

    xmin, xmax, nx = xmesh 
    if x_ref == np.array([]):
        x_ref = np.linspace(xmin,xmax,nx,endpoint=True)

    fnc1,fnc2,_,_ = nullclines(modelclass,xmesh,ymesh,t)
    g = lambda xx : fnc1(xx) - fnc2(xx)

    E = []
    for i in range(len(x_ref)):
        xij = fsolve(g,x0=x_ref[i])[0].round(sigfig)
        Eij = [xij,fnc1(xij).round(sigfig)]
        flag = inList(Eij,E)
        if not flag:
            vv = modelclass.f(np.array([Eij[0],Eij[1]]),t)
            if np.abs(vv[0]) <= tol and np.abs(vv[1]) <= tol:
                E.append(Eij)

    return np.array(E)


def equilibria3(modelclass,xmesh,ymesh,t=0,x_ref=np.array([]),tol=1e-3,sigfig=3):
    # equilibria2 but with fmin instead of fsolve

    xmin, xmax, nx = xmesh 
    # ymin, ymax, ny = ymesh
    if x_ref == np.array([]):
        x_ref = np.linspace(xmin,xmax,nx,endpoint=True)

    fnc1,fnc2,_,_ = nullclines(modelclass,xmesh,ymesh,t)
    g = lambda xx : np.abs(fnc1(xx) - fnc2(xx))

    E = []
    for i in range(len(x_ref)):
        xij = fmin(func=g, x0=x_ref[i])[0].round(sigfig)
        Eij = [xij,fnc1(xij).round(sigfig)]
        flag = inList(Eij,E)
        if not flag:
            vv = modelclass.f(np.array([Eij[0],Eij[1]]),t)
            if (np.abs(vv[0]) <= tol and np.abs(vv[1]) <= tol):
                E.append(Eij)

    E = np.array(E)

    return np.array(E)


def eigenvalsEquilibria(modelclass,E,sigfig=2):
    # extracts the eigenvalues L[i] corresponding to each equilibrium point E[i]

    L = []
    for i in range(len(E)):
        # Ei = E[i]
        Ji = modelclass.jacobian(E[i])
        # L.append(np.linalg.eigvals(Ji).round(sigfig))
        L.append(np.linalg.eigvals(Ji).round(sigfig))
        # print(E[i])
        # print(L)

    L = np.array(L)

    return L

def trajectories(modelclass,X0,tmesh,method="RK4"):
    # extracts the states X[t] for each time t involved in the time mesh T

    tend, dt = tmesh
    T = np.arange(0,tend+dt,dt)
    X = []
    for i in range(len(X0)):
        X.append(modelclass.timestepping(X0[i],dt,tend,method))

    return X,T #X is a list!

def trajectories2(modelclass,X0,tmesh,method="RK4"):
    # extracts the states X[t] for each time t involved in the time mesh T

    tend, dt = tmesh
    T = np.arange(0,tend+dt,dt)
    X = modelclass.timestepping(X0,dt,tend,method).T

    return X,T

def extractPhasePlaneAttributes(modelclass,xmesh,ymesh,t,eqtol):

    xmin, xmax, nx_pp, nx_nc, _ = xmesh 
    ymin, ymax, ny_pp, ny_nc, _ = ymesh

    xmesh_pp = xmin, xmax, nx_pp 
    ymesh_pp = ymin, ymax, ny_pp
    xmesh_nc = xmin, xmax, nx_nc 
    ymesh_nc = ymin, ymax, ny_nc

    u, v, X, Y = phasePlane(modelclass,xmesh_pp,ymesh_pp,t=t)
    fnc1, fnc2, x, _ = nullclines(modelclass,xmesh_nc,ymesh_nc,t=t) #,ncmethod)
    E = equilibria(modelclass,xmesh_pp,ymesh_pp,t=t,tol=eqtol)
    L = eigenvalsEquilibria(modelclass,E)

    return u,v,X,Y,fnc1,fnc2,x,E,L




