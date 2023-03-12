# main function ODE2D; an analysis tool two-dimensional dynamical systems
# Mike Wendels, last modified 2023/02/19

# ===== MODULES ===== #
import os
import numpy as np
import matplotlib.pyplot as plt
import ode2d_models as model
import ode2d_analysis as analysis
import ode2d_config as config
import ode2d_pplane as ode2d

import ode545_functions as functions

cwd = os.getcwd()

# ===== INPUT ===== #

# ----- model ----- #
case = "IZH-B"                                                  # case label; will be searched for in the ode2d_config file
savepath_main = cwd+"/IZH_model_B/"                             # main folder to save plots to; subfolders according to detailed settings will be made further below
fmodel, xmesh, ymesh, tmesh, X0, eqtol = config.getModel(case)  # gets (detailed) input regarding model, mesh, initial value and tolerance for finding equilibria from ode2d_config
tmesh = 1000.,0.01                                              # time mesh (different subfolders will be made for different time meshes)

# xmesh = -100.0, 50.0, 100, 500, "$v$"
# ymesh = -50.0, 300.0, 100, 500, "$u$"

# ----- parameters input current function fI ----- #
Iext = 100                                                      # input current
tthres = 500                                                     # threshold current
Pulses = [[180,200,Iext],[580,600,Iext]]                        # time intervals for current pulses 
Iref = 0                                                        # reference current
fI_lab = 2                                                      # choice for which current function (see output section for which label corresponds to which function)

# ----- plotting ----- #
opt_plot = 2                                                    # plotting option (1 = separate analysis plots, 2 = combined plots)
savedir = f"f{fI_lab}_tend={tmesh[0]}_dt={tmesh[1]}"            # name subfolder created for current run
savelabel = f"Iext={Iext}"                                      # string with which each file name in the subfolder starts
title = ""                                                      # title of the plots describing the case run (will be followed by some details about each plot specifically)


# ===== OUTPUT ===== #

savepath = savepath_main+savedir

if not os.path.exists(savepath_main):
    os.mkdir(savepath_main)

if not os.path.exists(savepath):
    os.mkdir(savepath)

if fI_lab == 1: # constant current
    fI = lambda t : model.fConstant(t,Iext)
elif fI_lab == 2: # Heaviside function
    fI = lambda t : model.fHeaviside(t,Iext,tthres)
elif fI_lab == 3: # linear interpolation between given points
    fI = lambda t : model.fPiecewiseLinear(t,[0,tthres,tmesh[0]],[0,Iext,Iext])
elif fI_lab == 4: # pulses of stength Pulses[2] for given time intervals (Pulses[0],Pulses[1])
    fI = lambda t : model.fPulses(t,Iref,Pulses)
else:
    print("ERROR [ode2d_main]: specify valid option current function fI_lab")
    exit(1)

### algorithm

Xtraj, T = functions.trajectories(fmodel(fI), X0, tmesh, method="RK4")
# Xtraj = Xtraj.T

# initialization summary plot
n = len(Xtraj[0])                                # dimensionality of the ODE
print(n)
nt = len(T)

Idata = np.zeros((1,nt))
for i in range(nt):
    Idata[0,i] = fI(T[i])
print(Idata)

for i in range(len(X0)):
    height_ratios = [2]+[1 for i in range(n+1)]
    print(height_ratios)
    fig, axs = plt.subplots(n+2, 1, figsize=(12, 12), height_ratios = height_ratios) #, gridspec_kw={'height_ratios': [8, 4, 4]})

    functions.plotPhasePlane(axs[0], fmodel(fI), xmesh, ymesh, Xtraj[i])
    functions.plotTimeSeries(axs[1:n+1], Xtraj[i], T)
    functions.plotTimeSeries(axs[n+1], Idata, T, Xlims = [-0.1*Iext, 1.1*Iext])

    fig.savefig(savepath+"/analysis_X0="+str(X0[i])+"-6", bbox_inches="tight")
    plt.close()




# print(X0)
# # print(fmodel)
# Xtraj, _ = ode2d.trajectories(fmodel(fI),X0,tmesh,method="RK4")
# for i in range(len(X0)):
#     savelabel_TSA_i = "TSA_X0="+str(X0[i])
#     savelabel_PPA_i = "PPA_X0="+str(X0[i])
#     print(Xtraj[i])
#     functions.TSA(Xtraj[i], tmesh, savepath, savelabel_TSA_i)
#     functions.PPA(fmodel(fI),xmesh,ymesh,Xtraj[i],savepath,savelabel_PPA_i)
#     functions.plotI(fI,tmesh, savepath, savelabel_TSA_i)

# plotI = [fI, "I", [0,Iext+10]] # information for plotting regarding input current function
# analysis.phasePlaneAnalysis(fmodel(fI),xmesh,ymesh,tmesh,X0=X0,t=tmesh[0],plot_traj=2,eqtol=eqtol,opt_plot=opt_plot,legend=False,title=title,plotI=plotI,savepath=savepath,savelabel=savelabel)
