# supporting functions for ode545_main
# Mike Wendels, last modified 3/17/2023

import numpy as np
import ode545_pplane as pplane


#####################
# GENERAL FUNCTIONS #
#####################

def trajectories(modelclass,X0,tmesh,method="RK4"):
    # extracts the states X[t] for each time t involved in the time mesh T

    tend, dt = tmesh
    T = np.arange(0,tend+dt,dt)
    X = modelclass.timestepping(X0,dt,tend,method).T

    return X,T

# def addWhiteNoise(x,eta):
#     # adds white noise with standard deviation eta to the input data
#     return x + np.random.normal(loc=0,scale=eta,size=x.shape)


##########################
# GENERAL PLOT FUNCTIONS #
##########################

def plotTimeSeries1d(ax, Xt, T, lab="", xlims=[], ylims=[], plot_traj=1):

    if plot_traj == 1:
        ax.plot(T,Xt,'k.-')
    elif plot_traj == 2:
        ax.plot(T,Xt,'k-')
    elif plot_traj == 3:
        ax.plot(T,Xt,'r-.')
    else:
        ax.plot(T[-1],Xt[-1],'k.')
    ax.set_xlabel("$t$")
    if not lab == "":
        ax.set_ylabel(lab)
    if not xlims==[]:
        ax.set_xlim(xlims)
    if not ylims==[]:
        ax.set_ylim(ylims)


###########################
# UMBRELLA PLOT FUNCTIONS #
###########################

def plotPhasePlane(ax, modelclass, xmesh, ymesh, Xtraj, t=10, plot_traj=1, eqtol=1e-3, legend=False, field_opt=1):
    # plots phase plane at time t for the ODE corresponding to modelclass with a vector field and nullclines all
    # determined by the given grids in xmesh, ymesh

    u, v, X, Y, fnc1, fnc2, x, E, L = pplane.extractPhasePlaneAttributes(modelclass,xmesh,ymesh,t,eqtol)

    xmin, xmax, _, _, xlab = xmesh 
    ymin, ymax, _, _, ylab = ymesh

    # plot vector field x[0]*
    if field_opt == 1:
        ax.streamplot(X, Y, u, v)
    else:
        ax.quiver(X, Y, u, v) #/np.linalg.norm(u), v/np.linalg.norm(u))
    ax.plot(x,fnc1(x),'k--',label="nullcline "+xlab)
    ax.plot(x,fnc2(x),'k-.',label="nullcline "+ylab)

    # plot state of system
    if not Xtraj == np.array([]):
        if plot_traj: # plots the complete trajectory in time
            ax.plot(Xtraj[0,:],Xtraj[1,:],'g.-',label = "traj. $x_1 =$"+str(Xtraj[0,0])+", $x_2 = $"+str(Xtraj[1,0]))
        else: # plots only the last point (state of system at the last time T[-1])
            ax.plot(Xtraj[0,-1],Xtraj[0,-1],'g.',label = "traj. $x_1 =$"+str(Xtraj[0,0])+", $x_2 = $"+str(Xtraj[1,0]))
        # Xtraj_end.append(np.array([Xi[-1,0],Xi[-1,1]]))

    # plot equilibria
    for i in range(len(E)):
        if np.all( L[i].real <= 0): # if all eigenvalues have non-positive real part, the equilibrium is stable (blue dot)
            ax.plot(E[i,0],E[i,1],'o',color='b',label="$x_1 =$"+str(E[i,0])+", $x_2 = $"+str(E[i,1])+", $\lambda_1=$"+str(L[i,0])+", $\lambda_2=$"+str(L[i,1])+")")
        else: # otherwise the equilibrium is unstable (red dot)
            ax.plot(E[i,0],E[i,1],'o',color='r',label="$x_1 =$"+str(E[i,0])+", $x_2 = $"+str(E[i,1])+", $\lambda_1=$"+str(L[i,0])+", $\lambda_2=$"+str(L[i,1])+")")

    # other plotting settings
    ax.set_xlim([xmin,xmax])
    ax.set_ylim([ymin,ymax])
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)

    if legend == 1:
        ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left",fontsize='small') #loc='lower right',fontsize='small')
    elif legend == 2:
        ax.legend(loc="upper left",fontsize='small')


def plotTimeSeries(axs, X, T, plot_traj = 1, Xlabs="", Xlims=[]):
    # plots the n given rows of Xtraj versus T on the n given axes
    
    n = len(X)                                  # dimensionality of the ODE
    tlims = [T[0],T[-1]]                        # time limits for plotting

    if Xlabs == "":
            Xlabs = ["" for i in range(n)]
    if Xlims == []:
            Xlims = [ [] for i in range(n)]

    if n > 1:
        for i in range(n):
            plotTimeSeries1d(axs[i], X[i], T, Xlabs[i], tlims, Xlims[i], plot_traj)
    else:
        plotTimeSeries1d(axs, X, T.reshape((1,len(T))), Xlabs, tlims, Xlims, plot_traj)
