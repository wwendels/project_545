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

import pysindy as ps

cwd = os.getcwd()

# ===== INPUT ===== #

# ----- model ----- #
case = "FHN-A" #"IZH-B"                                                  # case label; will be searched for in the ode2d_config file
savepath_main = cwd+"/FHN_model_A/"                             # main folder to save plots to; subfolders according to detailed settings will be made further below
fmodel, xmesh, ymesh, tmesh, X0, eqtol = config.getModel(case)  # gets (detailed) input regarding model, mesh, initial value and tolerance for finding equilibria from ode2d_config
tmesh = 1000., 0.01 #05 #1                                              # time mesh (different subfolders will be made for different time meshes)

# xmesh = -100.0, 50.0, 100, 500, "$v$"
# ymesh = -50.0, 300.0, 100, 500, "$u$"

# ----- parameters input current function fI ----- #
Iext = 0.04 #100                                                      # input current
tthres = 500                                                     # threshold current
Pulses = [[180,200,Iext],[580,600,Iext]]                        # time intervals for current pulses 
Iref = 0                                                        # reference current
fI_lab = 1                                                      # choice for which current function (see output section for which label corresponds to which function)

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

Xtraj, T = functions.trajectories(fmodel(fI), X0, tmesh, method="RK4") # list of arrays of ndims x nt
print(Xtraj[0].shape)
dt = T[1] - T[0]
# Xtraj = Xtraj.T

# initialization summary plot
n = len(Xtraj[0])                                # dimensionality of the ODE
print(n)
nt = len(T)

Idata = np.zeros((1,nt))
for i in range(nt):
    Idata[0,i] = fI(T[i])
print(Idata)

models = []

for i in range(len(X0)):
    height_ratios = [2]+[1 for i in range(n+1)]
    print(height_ratios)
    fig, axs = plt.subplots(n+2, 1, figsize=(12, 12), height_ratios = height_ratios) #, gridspec_kw={'height_ratios': [8, 4, 4]})

    functions.plotPhasePlane(axs[0], fmodel(fI), xmesh, ymesh, Xtraj[i])
    functions.plotTimeSeries(axs[1:n+1], Xtraj[i], T)
    functions.plotTimeSeries(axs[n+1], Idata, T, Xlims = [-0.1*Iext, 1.1*Iext])

    fig.savefig(savepath+"/analysis_X0="+str(X0[i]).replace(" ","")+"_"+savelabel+"-FHN-A-1.png", bbox_inches="tight")
    plt.close()


####################
# PYSINDY RECOVERY #
####################

# # ----- training parameters ----- #

# # general
# X0_train = [np.array([0, 0.15])]
# I_train = [0, 0.1]
# # Eta = [0]
# eta = 0

# # SINDy
# poly_order = 3          # highest polynomial degree
# threshold = 0.005       # STLSQ optimizer threshold (should be >= smallest coefficient in model)

# X_train = []
# for X0i in X0_train:
#     for Ii in I_train:
#         fIi = lambda t : model.fConstant(t,Ii)
#         X_train_i, T = functions.trajectories(fmodel(fIi), X0i, tmesh, method="RK4")
#         X_train.append(X_train_i.T)








# train model for first X0
poly_order = 3
threshold = 0.005
model_ps = ps.SINDy(optimizer=ps.STLSQ(threshold=threshold), feature_library=ps.PolynomialLibrary(degree=poly_order))
# model_ps = ps.SINDy(feature_library=ps.PolynomialLibrary(degree=poly_order))
print(Xtraj[0].T)
model_ps.fit(Xtraj[0].T, t=dt)
model_ps.print()

# assert len(X0) > 1

X0_test = [np.array([0, 0.2]), np.array([0, 0.25]), np.array([0.5, 0.2])]
Xtest_ref, _ = functions.trajectories(fmodel(fI), X0_test, tmesh, method="RK4") # list of arrays of ndims x nt

height_ratios = [2]+[1 for i in range(n)]
fig, axs = plt.subplots(n+1, 1, figsize=(12, 12), height_ratios = height_ratios) #, gridspec_kw={'height_ratios': [8, 4, 4]})

for i in range(len(X0_test)):
    Xtest_i = model_ps.simulate(X0_test[i], T).T

    functions.plotPhasePlane(axs[0], fmodel(fI), xmesh, ymesh, Xtest_ref[i])
    functions.plotTimeSeries(axs[1:n+1], Xtest_ref[i], T) #, plot_traj=2)
    functions.plotTimeSeries(axs[1:n+1], Xtest_i, T, plot_traj=3)
    # functions.plotTimeSeries(axs[n+1], Idata, T, Xlims = [-0.1*Iext, 1.1*Iext])

fig.savefig(savepath+"/verification-FHN-A-1"+savelabel+".png", bbox_inches="tight")
plt.close()






# Xtraj_comp = model_ps.simulate(X0[1], T)
# fig, axs = plt.subplots(n, 1, sharex=True, figsize=(7, 9))
# for i in range(n):
#     axs[i].plot(T, Xtraj_comp[:, i], 'k', label='numerical derivative')
# fig.savefig(savepath+"/prediction_X0="+str(X0[i]).replace(".","")+"-FHN-A-1.png", bbox_inches="tight")
# plt.close()



