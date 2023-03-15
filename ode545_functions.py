import numpy as np
import matplotlib.pyplot as plt
import ode2d_analysis as analysis
import ode2d_pplane as ode2d


def trajectories(modelclass,X0,tmesh,method="RK4"):
    # extracts the states X[t] for each time t involved in the time mesh T

    tend, dt = tmesh
    T = np.arange(0,tend+dt,dt)
    X = []
    for i in range(len(X0)):
        X.append(modelclass.timestepping(X0[i],dt,tend,method).T)

    return X,T #X is a list!

def trajectories2(modelclass,X0,tmesh,method="RK4"):
    # extracts the states X[t] for each time t involved in the time mesh T

    tend, dt = tmesh
    T = np.arange(0,tend+dt,dt)
    X = modelclass.timestepping(X0,dt,tend,method).T

    return X,T #X is a list!

def addWhiteNoise(x,eta):
    return x + np.random.normal(loc=0,scale=eta,size=x.shape)



##########################
# GENERAL PLOT FUNCTIONS #
##########################

def plotTimeSeries1d(ax, Xt, T, lab="", xlims=[], ylims=[], plot_traj=1):

    if plot_traj == 1:
        ax.plot(T,Xt,'k.-')
    elif plot_traj == 2:
        ax.plot(T,Xt,'k-')
    elif plot_traj == 3:
        ax.plot(T,Xt,'r-')
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

    # u, v, X, Y, fnc1, fnc2, x, E, L = analysis.extractPhasePlaneAttributes(modelclass,xmesh,ymesh,t,eqtol)
    # analysis.plotPhasePlane(ax, u, v, X, Y, x, fnc1, fnc2, E, L, Xtraj, xmesh, ymesh, legend, "", plot_traj)

    u, v, X, Y, fnc1, fnc2, x, E, L = analysis.extractPhasePlaneAttributes(modelclass,xmesh,ymesh,t,eqtol)

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
        # L = np.linalg.eigvals(E) #np.linalg.eigvalsh for hermitian
        # ax.legend()

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
        # print("AXESS")
        # print(len(axs))
        
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






# def TSA(Xtraj, tmesh, savepath, savelabel, plot_traj = 1, Xlabs="", Xlims="", plotI=""):
        
#         # print(Xtraj)
#         n = len(Xtraj.T) #dimensionality of the ODE
#         # print(n)
#         T = np.arange(0,tmesh[0]+tmesh[1],tmesh[1])
#         tlims = [0,tmesh[0]]

#         if plotI == "":
#             fig, axs = plt.subplots(n, 1, figsize=(12, 12)) #, gridspec_kw={'height_ratios': [8, 4, 4]})
#         else:
#             fig, axs = plt.subplots(n+1, 1, figsize=(10, 16)) #, gridspec_kw={'height_ratios': [8, 4, 4, 4]})
#         savename = savepath+"/"+savelabel+".png"

#         if Xlabs == "":
#              Xlabs = ["" for i in range(n)]
#         if Xlims == "":
#              Xlims = [ [] for i in range(n)]

#         for i in range(n):
#             print(Xtraj[:,i])
#             plotTimeSeries(axs[i], Xtraj[:,i], T, Xlabs[i], tlims, Xlims[i], plot_traj)

#         if not plotI == "":
#             fIext,Ilab,Ilims = plotI[0],plotI[1],plotI[2]
#             nt = len(T)
#             Iext = np.zeros(nt)
#             for i in range(nt):
#                 Iext[i] = fIext(T[i])
#             plotTimeSeries(axs[n+1], Iext, T, Ilab, tlims, Ilims, plot_traj)

#         fig.savefig(savename,bbox_inches="tight")
#         plt.close()


# def PPA(modelclass,xmesh,ymesh,Xtraj,savepath,savelabel,t=0,plot_traj=1,eqtol=1e-3,legend=True,title=""):

#     # if not X0 == []:
#     #     Xtraj,_ = ode2d.trajectories(modelclass,X0,tmesh,method="RK4")
#     # else:
#     #     Xtraj,_ = [],[]
#     # Xtraj = Xtraj[0]

#     u,v,X,Y,fnc1,fnc2,x,E,L = analysis.extractPhasePlaneAttributes(modelclass,xmesh,ymesh,t,eqtol)
    

#     # plot phase plane
#     fig = plt.figure()
#     savename = savepath+"/"+savelabel+"_pplane_analysis.png"
#     analysis.plotPhasePlane(plt.gca(),u,v,X,Y,x,fnc1,fnc2,E,L,Xtraj,xmesh,ymesh,legend,title,plot_traj)
#     fig.savefig(savename,bbox_inches="tight")
#     plt.close()

# def plotI(ax, fI, tmesh, plot_traj = 1):
        
#         tlims = [0,tmesh[0]]
#         T = np.arange(0,tmesh[0]+tmesh[1],tmesh[1])
#         nt = len(T)

#         Iext = np.zeros(nt)
#         for i in range(nt):
#             Iext[i] = fI(T[i])

#         plotTimeSeries(ax, Iext, T, "$I(t)$", tlims, Ilims, plot_traj)
 
#         # fig.savefig(savename,bbox_inches="tight")
#         # plt.close()
