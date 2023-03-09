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

# ----- parameters input current function fI ----- #
Iext = 100                                                      # input current
tthres = 50                                                     # threshold current
Pulses = [[180,200,Iext],[580,600,Iext]]                        # time intervals for current pulses 
Iref = 0                                                        # reference current
fI_lab = 2                                                      # choice for which current function (see output section for which label corresponds to which function)

# ----- plotting ----- #
opt_plot = 2                                                    # plotting option (1 = separate analysis plots, 2 = combined plots)
savedir = f"f{fI_lab}_tend={tmesh[0]}_dt={tmesh[1]}"            # name subfolder created for current run
savelabel = f"Iext={Iext}"                                      # string with which each file name in the subfolder starts
title = ""                                                      # title of the plots describing the case run (will be followed by some details about each plot specifically)


# ===== OUTPUT ===== #

savepath = savepath_main+savedir+"bla"

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

print(X0)
Xtraj = ode2d.trajectories(fmodel,X0,tmesh,method="RK4")

functions.TSA(Xtraj, tmesh, savepath)

# plotI = [fI, "I", [0,Iext+10]] # information for plotting regarding input current function
# analysis.phasePlaneAnalysis(fmodel(fI),xmesh,ymesh,tmesh,X0=X0,t=tmesh[0],plot_traj=2,eqtol=eqtol,opt_plot=opt_plot,legend=False,title=title,plotI=plotI,savepath=savepath,savelabel=savelabel)
