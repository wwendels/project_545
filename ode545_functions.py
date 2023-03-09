import numpy as np
import matplotlib.pyplot as plt
# import ode2d_analysis as analysis

def plotTimeSeries(ax,Xt,T,lab="",xlims=[],ylims=[],plot_traj=1):

    if plot_traj == 1:
        ax.plot(T,Xt,'k.-')
    elif plot_traj == 2:
        ax.plot(T,Xt,'k-')
    else:
        ax.plot(T[-1],Xt[-1],'k.')
    ax.set_xlabel("$t$")
    if not lab == "":
        ax.set_ylabel(lab)
    if not xlims==[]:
        ax.set_xlim(xlims)
    if not ylims==[]:
        ax.set_ylim(ylims)

def TSA(Xtraj, tmesh, savepath, plot_traj = 1, Xlabs="", Xlims="", plotI=""):
        
        n = len(Xtraj.T) #dimensionality of the ODE
        T = np.arange(0,tmesh[0]+tmesh[1],tmesh[1])
        tlims = [0,tmesh[0]]

        if plotI == "":
            fig, axs = plt.subplots(n, 1, figsize=(12, 12), gridspec_kw={'height_ratios': [8, 4, 4]})
        else:
            fig, axs = plt.subplots(n+1, 1, figsize=(10, 16), gridspec_kw={'height_ratios': [8, 4, 4, 4]})
        savename = savepath+".png"

        if Xlabs == "":
             Xlabs = ["" for i in range(n)]
        if Xlims == "":
             Xlims = ["" for i in range(n)]

        for i in range(n):
            plotTimeSeries(axs[i], Xtraj.T[i], T, Xlabs[i], tlims, Xlims[i], plot_traj)

        if not plotI == "":
            fIext,Ilab,Ilims = plotI[0],plotI[1],plotI[2]
            nt = len(T)
            Iext = np.zeros(nt)
            for i in range(nt):
                Iext[i] = fIext(T[i])
            plotTimeSeries(axs[n+1], Iext, T, Ilab, tlims, Ilims, plot_traj)

        fig.savefig(savename,bbox_inches="tight")
        plt.close()

