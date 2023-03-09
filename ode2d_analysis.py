# phase plane plotting functions for analysis 2D ODE systems
# Mike Wendels, last modified 2/19/2023

import numpy as np
import matplotlib.pyplot as plt
import imageio # for gifs
import ode2d_pplane as ode2d

def plotPhasePlane(ax,u,v,X,Y,x,fnc1,fnc2,E,L,Xtraj,xmesh,ymesh,legend,title,plot_traj=1):

    xmin, xmax, _, _, xlab = xmesh 
    ymin, ymax, _, _, ylab = ymesh

    ax.quiver(X, Y, u/np.linalg.norm(u), v/np.linalg.norm(v))
    # ax.streamplot(X, Y, u, v)
    ax.plot(x,fnc1(x),'r',label="nullcline "+xlab)
    ax.plot(x,fnc2(x),'b',label="nullcline "+ylab)

    if not Xtraj == np.array([]):
        if plot_traj >= 1:
            ax.plot(Xtraj[:,0],Xtraj[:,1],'g.-',label = "traj. $x_1 =$"+str(Xtraj[0,0])+", $x_2 = $"+str(Xtraj[0,1]))
        else:
            ax.plot(Xtraj[-1,0],Xtraj[-1,1],'g.',label = "traj. $x_1 =$"+str(Xtraj[0,0])+", $x_2 = $"+str(Xtraj[0,1]))
        # Xtraj_end.append(np.array([Xi[-1,0],Xi[-1,1]]))

    for i in range(len(E)):
        ax.plot(E[i,0],E[i,1],'o',color='black',label="$x_1 =$"+str(E[i,0])+", $x_2 = $"+str(E[i,1])+"$\lambda_1=$"+str(L[i,0])+", $\lambda_2=$"+str(L[i,1])+")")
        # L = np.linalg.eigvals(E) #np.linalg.eigvalsh for hermitian
        # ax.legend()

    ax.set_xlim([xmin,xmax])
    ax.set_ylim([ymin,ymax])
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)

    if legend == 1:
        ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left",fontsize='small') #loc='lower right',fontsize='small')
    elif legend == 2:
        ax.legend(loc="upper left",fontsize='small')
    if title:
        ax.set_title(title)


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


def extractPhasePlaneAttributes(modelclass,xmesh,ymesh,t,eqtol):

    xmin, xmax, nx_pp, nx_nc, _ = xmesh 
    ymin, ymax, ny_pp, ny_nc, _ = ymesh

    xmesh_pp = xmin, xmax, nx_pp 
    ymesh_pp = ymin, ymax, ny_pp
    xmesh_nc = xmin, xmax, nx_nc 
    ymesh_nc = ymin, ymax, ny_nc

    u, v, X, Y = ode2d.phasePlane(modelclass,xmesh_pp,ymesh_pp,t=t)
    fnc1, fnc2, x, _ = ode2d.nullclines(modelclass,xmesh_nc,ymesh_nc,t=t) #,ncmethod)
    E = ode2d.equilibria(modelclass,xmesh_pp,ymesh_pp,t=t,tol=eqtol)
    L = ode2d.eigenvalsEquilibria(modelclass,E)

    return u,v,X,Y,fnc1,fnc2,x,E,L

# def extractTrajectory(modelclass,X0,tmesh):
#     Xtraj,T = ode2d.trajectories(modelclass,X0,tmesh,method="RK4")
#     return Xtraj,T

def phasePlaneAnalysis(modelclass,xmesh,ymesh,tmesh,X0,t=0,plot_traj=1,eqtol=1e-3,opt_plot=1,legend=True,title="",x1lims=[],x2lims=[],plotI="",savepath="",savelabel=""):

    if not X0 == []:
        Xtraj,T = ode2d.trajectories(modelclass,X0,tmesh,method="RK4")
    else:
        Xtraj,T = [],[]
    tlims = [T[0],T[-1]]
    Xtraj = Xtraj[0]

    u,v,X,Y,fnc1,fnc2,x,E,L = extractPhasePlaneAttributes(modelclass,xmesh,ymesh,t,eqtol)

    xmin, xmax, _, _, xlab = xmesh 
    ymin, ymax, _, _, ylab = ymesh
    if x1lims == []:
        x1lims = [xmin,xmax]
    if x2lims == []:
        x2lims = [ymin,ymax]
    # tlims = [0,tend]

    if opt_plot == 0:

        # plot phase plane
        fig = plt.figure()
        savename = savepath+savelabel+"_pplane_analysis.png"
        plotPhasePlane(plt.gca(),u,v,X,Y,x,fnc1,fnc2,E,L,Xtraj,xmesh,ymesh,legend,title,plot_traj)
        fig.savefig(savename,bbox_inches="tight")
        plt.close()

        # plot time series x1
        fig = plt.figure()
        plotTimeSeries(plt.gca(),Xtraj.T[0],T,xlab,tlims,x1lims,plot_traj)
        if title:
            plt.title(title+" $x_0=($"+str(Xtraj[0,0])+"$)$")
        fig.savefig(savepath+savelabel+"_time_series_x1_X0="+str(Xtraj[0,:])+".png")
        plt.close()

        # plot time series x2
        fig = plt.figure()
        plotTimeSeries(plt.gca(),Xtraj.T[1],T,ylab,tlims,x2lims,plot_traj)
        if title:
            plt.title(title+" $x_0=($"+str(Xtraj[0,1])+"$)$")
        fig.savefig(savepath+savelabel+"_time_series_x2_X0="+str(Xtraj[0,:])+".png")
        plt.close()

        # plot time series x1
        if not plotI == "":
            fig = plt.figure()
            savename = savepath+savelabel+"_It.png"

            fIext,Ilab,Ilims = plotI[0],plotI[1],plotI[2]
            nt = len(T)
            Iext = np.zeros(nt)
            for i in range(nt):
                Iext[i] = fIext(T[i])
            plotTimeSeries(fig.gca(),Iext,T,Ilab,tlims,Ilims,plot_traj)

            fig.savefig(savename,bbox_inches="tight")
            plt.close() 

    if opt_plot == 1:

        # phase plane
        fig, (ax1,ax2,ax3) = plt.subplots(3,1,figsize=(12, 12),gridspec_kw={'height_ratios': [8, 4, 4]}) #2,1
        savename = savepath+savelabel+".png" #analysis.png
        plotPhasePlane(ax1,u,v,X,Y,x,fnc1,fnc2,E,L,Xtraj,xmesh,ymesh,legend,title,plot_traj)
        plotTimeSeries(ax2,Xtraj.T[0],T,xlab,tlims,x1lims,plot_traj)
        plotTimeSeries(ax3,Xtraj.T[1],T,ylab,tlims,x2lims,plot_traj)
        fig.savefig(savename,bbox_inches="tight")
        plt.close()

        # # plot time series x1
        # if not plotI == "":
        #     fig = plt.figure()
        #     savename = savepath+label+"_It.png"

        #     fIext,Ilab,Ilims = plotI[0],plotI[1],plotI[2]
        #     nt = len(T)
        #     Iext = np.zeros(nt)
        #     for i in range(nt):
        #         Iext[i] = fIext(T[i])
        #     plotTimeSeries(fig.gca(),Iext,T,Ilab,tlims,Ilims,plot_traj)

        #     fig.savefig(savename,bbox_inches="tight")
        #     plt.close()        

    if opt_plot == 2:

        # phase plane + plot time series x1
        if plotI == "":
            fig, (ax1,ax2,ax3) = plt.subplots(3,1,figsize=(12, 12),gridspec_kw={'height_ratios': [8, 4, 4]}) #2,1
        else:
            fig, (ax1,ax2,ax3,ax4) = plt.subplots(4,1,figsize=(10, 16),gridspec_kw={'height_ratios': [8, 4, 4, 4]}) #2,1
        savename = savepath+savelabel+".png" #analysis.png
        plotPhasePlane(ax1,u,v,X,Y,x,fnc1,fnc2,E,L,Xtraj,xmesh,ymesh,legend,title,plot_traj)
        plotTimeSeries(ax2,Xtraj.T[0],T,xlab,tlims,x1lims,plot_traj)
        plotTimeSeries(ax3,Xtraj.T[1],T,ylab,tlims,x2lims,plot_traj)
        if not plotI == "":
            fIext,Ilab,Ilims = plotI[0],plotI[1],plotI[2]
            nt = len(T)
            Iext = np.zeros(nt)
            for i in range(nt):
                Iext[i] = fIext(T[i])
            plotTimeSeries(ax4,Iext,T,Ilab,tlims,Ilims,plot_traj)
        fig.savefig(savename,bbox_inches="tight")
        plt.close()

    return savename


def phasePlaneAnalysis2(modelclass,xmesh,ymesh,t=0,Xtraj=np.array([]),T=np.array([]),plot_traj=1,label="",eqtol=1e-3,opt_plot=2,legend=True,title="",x1lims=[],x2lims=[],tlims=[],plotI="",savepath=""):
    # specially made for GIFs in time, as it uses a predetermined Xtraj

    u,v,X,Y,fnc1,fnc2,x,E,L = extractPhasePlaneAttributes(modelclass,xmesh,ymesh,t,eqtol)

    xmin, xmax, _, _, xlab = xmesh 
    ymin, ymax, _, _, ylab = ymesh
    if x1lims == []:
        x1lims = [xmin,xmax]
    if x2lims == []:
        x2lims = [ymin,ymax]
    # tlims = [0,tend]

    if opt_plot == 0 or opt_plot==2:

        # plot phase plane
        fig = plt.figure()
        savename = savepath+label+"_pplane_analysis.png"
        plotPhasePlane(plt.gca(),u,v,X,Y,x,fnc1,fnc2,E,L,Xtraj,xmesh,ymesh,legend,title,plot_traj)
        fig.savefig(savename,bbox_inches="tight")
        plt.close()

        # plot time series x1
        fig = plt.figure()
        plotTimeSeries(plt.gca(),Xtraj.T[0],T,xlab,tlims,x1lims,plot_traj)
        if title:
            plt.title(title+" $x_0=($"+str(Xtraj[0,0])+"$)$")
        fig.savefig(savepath+label+"_time_series_x1_X0="+str(Xtraj[0,:])+".png")
        plt.close()

        # plot time series x2
        fig = plt.figure()
        plotTimeSeries(plt.gca(),Xtraj.T[1],T,ylab,tlims,x2lims,plot_traj)
        if title:
            plt.title(title+" $x_0=($"+str(Xtraj[0,1])+"$)$")
        fig.savefig(savepath+label+"_time_series_x2_X0="+str(Xtraj[0,:])+".png")
        plt.close()

    if opt_plot == 1 or opt_plot == 2:

        # phase plane + plot time series x1
        if plotI == "":
            fig, (ax1,ax2,ax3) = plt.subplots(3,1,figsize=(12, 12),gridspec_kw={'height_ratios': [8, 4, 4]}) #2,1
        else:
            fig, (ax1,ax2,ax3,ax4) = plt.subplots(4,1,figsize=(10, 16),gridspec_kw={'height_ratios': [8, 4, 4, 4]}) #2,1
        savename = savepath+label+".png" #"_analysis.png"
        plotPhasePlane(ax1,u,v,X,Y,x,fnc1,fnc2,E,L,Xtraj,xmesh,ymesh,legend,title,plot_traj)
        plotTimeSeries(ax2,Xtraj.T[0],T,xlab,tlims,x1lims,plot_traj)
        plotTimeSeries(ax3,Xtraj.T[1],T,ylab,tlims,x2lims,plot_traj)
        # if not plotI == "":
        #     fIext,Ilab,Ilims = plotI[0],plotI[1],plotI[2]
        #     plotTimeSeries(ax4,fIext(T),T,Ilab,tlims,Ilims,plot_traj)
        # fig.savefig(savename,bbox_inches="tight")
        if not plotI == "":
            fIext,Ilab,Ilims = plotI[0],plotI[1],plotI[2]
            nt = len(T)
            Iext = np.zeros(nt)
            for i in range(nt):
                Iext[i] = fIext(T[i])
            plotTimeSeries(ax4,Iext,T,Ilab,tlims,Ilims,plot_traj)
        fig.savefig(savename,bbox_inches="tight")
        plt.close()

    return savename

def bifurcationAnalysisI(modelclassI,label,I,xmesh,ymesh,tmesh,X0,t=0,eqtol=1e-3,opt_plot=1,x1lims=[],x2lims=[],savepath=""): #ncmethod=1

    filenames1 = []
    j = 0
    for Ii in I:
        modelclass_i = modelclassI(Ii)
        j += 1
        label_i = label+"-"+str(j) #str(T[i])
        # label_i = label+str(Ii)
        title_i = "$I = $"+str(Ii)
        savename1 = phasePlaneAnalysis(modelclass_i,label_i,xmesh,ymesh,tmesh,X0,t=t,plot_traj=2,eqtol=eqtol,opt_plot=opt_plot,legend=False,title=title_i,x1lims=x1lims,x2lims=x2lims,savepath=savepath)
        filenames1.append(savename1)

    # build gif
    gifname = savepath+label+"_bifurcation_analysis.gif"
    with imageio.get_writer(gifname, mode='I') as writer:
        for filename in filenames1:
            image = imageio.imread(filename)
            writer.append_data(image)
    
    # Remove files
    # for filename in set(filenames1):
    #     os.remove(filename)

def bifurcationAnalysisTime(modelclass,label,xmesh,ymesh,tmesh,X0,eqtol=1e-3,opt_plot=1,x1lims=[],x2lims=[],plotI="",savepath=""): #ncmethod=1

    if not X0 == []:
        Xtraj,T = ode2d.trajectories(modelclass,X0,tmesh,method="RK4")
    else:
        Xtraj,T = [],[]
    tlims = [T[0],T[-1]]
    Xtraj = Xtraj[0]

    filenames1 = []
    # tend, dt = tmesh
    # nt = int(tend/dt)
    # T = np.arange(0,tend+dt,dt)
    j = 0
    for i in range(len(T)):
        j += 1
        label_i = label+"-"+str(j) #str(T[i])
        title_i = "$t = $"+str(T[i])
        savename1 = phasePlaneAnalysis2(modelclass,xmesh,ymesh,t=T[i],Xtraj=Xtraj[:i+1],T=T[:i+1],plot_traj=2,label=label_i,eqtol=eqtol,opt_plot=opt_plot,legend=False,title=title_i,x1lims=x1lims,x2lims=x2lims,tlims=tlims,plotI=plotI,savepath=savepath)
        filenames1.append(savename1)

    # build gif
    gifname = savepath+label+"_time_analysis.gif"
    with imageio.get_writer(gifname, mode='I') as writer:
        for filename in filenames1:
            image = imageio.imread(filename)
            writer.append_data(image)
    
    # Remove files
    # for filename in set(filenames1):
    #     os.remove(filename)


