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
case = "FHN-A"                                                  # case label; will be searched for in the ode2d_config file
savepath_main = cwd+"/FHN_model_A/"                             # main folder to save plots to; subfolders according to detailed settings will be made further below
fmodel, xmesh, ymesh, tmesh, X0, eqtol = config.getModel(case)  # gets (detailed) input regarding model, mesh, initial value and tolerance for finding equilibria from ode2d_config
tmesh = 1000., 0.01                                             # time mesh (different subfolders will be made for different time meshes)
seed = 37

# ----- parameters input current function fI ----- #
Iext = 0.04 #100                                                      # input current
tthres = 500                                                     # threshold current
Pulses = [[180,200,Iext],[580,600,Iext]]                        # time intervals for current pulses 
Iref = 0                                                        # reference current
fI_lab = 1                                                      # choice for which current function (see output section for which label corresponds to which function)

# ----- SINDy training parameters ----- #
# NOTE: applying SINDy only possible if model_recovery == True and fI_lab == 1 

model_recovery = True   # to apply SINDy

# general training parameters
train_opt = 1           # specifies X0 and I for training data (see below)
test_opt = 1            # specifies X0 and I for test data (see below)
eta = 0                 # std Gaussian noise applied

# SINDy specific training parameters
poly_order = 3          # highest polynomial degree
threshold = 0.005       # STLSQ optimizer threshold (should be >= smallest coefficient in model)


if model_recovery:
    savedir = f"SINDy_" + case + f"_eta={eta}_poly-order={poly_order}_thr={threshold}_seed={seed}"  # name subfolder created for current run
    savelabel = f"train-opt={train_opt}_test-opt={test_opt}"                            # string with which each file name in the subfolder starts
else:
    savedir = f"f{fI_lab}_tend={tmesh[0]}_dt={tmesh[1]}"            # name subfolder created for current run
    savelabel = f"Iext={Iext}"                                      # string with which each file name in the subfolder starts


# ===== OUTPUT ===== #

np.random.seed(seed)

savepath = savepath_main+savedir

if not os.path.exists(savepath_main):
    os.mkdir(savepath_main)

if not os.path.exists(savepath):
    os.mkdir(savepath)

# ----- current functions ----- #

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

# ----- PySINDy training/test data ----- #

if train_opt == 1:
    X0_train = [np.array([0, 0.2])]
    I_train = [0]
elif train_opt == 2:
    X0_train = [np.array([0, 0.2]), np.array([0.4, 0]), np.array([0.8, 0.1])]
    I_train = [0]
elif train_opt == 3:
    X0_train = [np.array([0, 0.2])]
    I_train = [0, 0.01, 0.02, 0.03]
elif train_opt == 4:
    X0_train = [np.array([0, 0.2])]
    I_train = [0, 0.02, 0.04, 0.06]
elif train_opt == 5:
    X0_train = [np.array([0, 0.2]), np.array([0.4, 0]), np.array([0.8, 0.1])]
    I_train = [0, 0.01, 0.02, 0.03]
elif train_opt == 6:
    X0_train = [np.array([0, 0.2]), np.array([0.4, 0]), np.array([0.8, 0.1])]
    I_train = [0, 0.02, 0.04, 0.06]

if test_opt == 1:
    X0_test = [np.array([0, 0.1]), np.array([0.4, 0.2]), np.array([0.8, 0])]
    I_test = [0, 0.01, 0.02, 0.03]
elif test_opt == 2:
    X0_test = [np.array([0, 0.1]), np.array([0.4, 0.2]), np.array([0.8, 0])]
    I_test = [0, 0.02, 0.04, 0.06]


##########################
# GENERAL MODEL ANALYSIS #
##########################
# gives phase plane and time series for the specified fI and X0

# trajectories state of system
Xtraj, T = functions.trajectories(fmodel(fI), X0, tmesh, method="RK4") # list of arrays of ndims x nt ( see print(Xtraj[0].shape) )

# initialization summary plot
n, nt = Xtraj[0].shape                              # n: dimensionality of the ODE, nt: number of time steps ( = len(T) )
dt = T[1] - T[0]                                    # Delta t

# trajectory input current
Idata = np.zeros((1,nt))
for i in range(nt):
    Idata[0,i] = fI(T[i])


# plot phase plane, time series state of system and time series input current
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
# extracts the equations for the dynamical system according to synthetic data generated from it, possibly with noise added

if model_recovery:

    assert fI_lab == 1 # only possible/logical to do this with a constant input function (with strength I as specified below)

    # X0_train = [np.array([0, 0.15])]
    # I_train = [0.04] 
    # eta = 0
    # X0_test = [np.array([0, 0.2]), np.array([0, 0.25])] #, np.array([0.5, 0.2])]
    # I_test = [0] #, 0.01]

    # ----- model training ----- #

    # training
    X_train = np.array([], dtype=np.int64).reshape(0,n)

    for X0i in X0_train:
        for Ii in I_train:
            fIi = lambda t : model.fConstant(t,Ii)
            X_train_i, T = functions.trajectories2(fmodel(fIi), X0i, tmesh, method="RK4")
            X_train_i += + np.random.normal(loc=0, scale=eta, size=X_train_i.shape) #= functions.addWhiteNoise(X_train_i, eta)
            X_train = np.vstack([X_train, X_train_i.T])

    model_ps = ps.SINDy(optimizer=ps.STLSQ(threshold=threshold), feature_library=ps.PolynomialLibrary(degree=poly_order))
    model_ps.fit(X_train, t=dt)

    # save model in a text file (and print on the command line)
    # NOTE: model_ps.print() does the same but cannot be used to obtain the string for the text file as it only prints and does not return a string
    model_txt = open(savepath+"model.txt","w")
    eqns = model_ps.equations(precision=3)
    feature_names = model_ps.feature_names
    for i, eqn in enumerate(eqns):
        names = "(" + feature_names[i] + ")"
        print(names + "' = " + eqn)
        model_txt.write(names + " = " + eqn +" \n")
    model_txt.close()

    # ----- plot of reference cases (true vs. approximated) ----- #
    # height_ratios = [2]+[1 for i in range(n)]
    # fig, axs = plt.subplots(n+1, 1, figsize=(12, 12), height_ratios = height_ratios) #, gridspec_kw={'height_ratios': [8, 4, 4]})
    fig, axs = plt.subplots(n, 1, figsize=(12, 12)) #, gridspec_kw={'height_ratios': [8, 4, 4]})

    for X0i in X0_test:
        for Ii in I_test:
            # print(X0i, Ii)
            fIi = lambda t : model.fConstant(t,Ii)
            X_test_i_ref, T = functions.trajectories2(fmodel(fIi), X0i, tmesh, method="RK4")
            X_test_i_sim = model_ps.simulate(X0i, T).T

            # functions.plotPhasePlane(axs[0], fmodel(fI), xmesh, ymesh, X_test_i_ref)
            # functions.plotTimeSeries(axs[1:n+1], X_test_i_sim, T) #, plot_traj=2)
            # functions.plotTimeSeries(axs[1:n+1], X_test_i_ref, T, plot_traj=3)
            # functions.plotTimeSeries(axs[n+1], Idata, T, Xlims = [-0.1*Iext, 1.1*Iext])
            functions.plotTimeSeries(axs, X_test_i_sim, T) #, plot_traj=2)
            functions.plotTimeSeries(axs, X_test_i_ref, T, plot_traj=3)

    fig.savefig(savepath+"/model-verification-FHN-A-1"+savelabel+".png", bbox_inches="tight")
    plt.close()


#############################################################

# # train model for first X0
# poly_order = 3
# threshold = 0.005
# model_ps = ps.SINDy(optimizer=ps.STLSQ(threshold=threshold), feature_library=ps.PolynomialLibrary(degree=poly_order))
# # model_ps = ps.SINDy(feature_library=ps.PolynomialLibrary(degree=poly_order))
# print(Xtraj[0].T)
# model_ps.fit(Xtraj[0].T, t=dt)
# model_ps.print()

# # assert len(X0) > 1

# X0_test = [np.array([0, 0.2]), np.array([0, 0.25]), np.array([0.5, 0.2])]
# Xtest_ref, _ = functions.trajectories(fmodel(fI), X0_test, tmesh, method="RK4") # list of arrays of ndims x nt

# height_ratios = [2]+[1 for i in range(n)]
# fig, axs = plt.subplots(n+1, 1, figsize=(12, 12), height_ratios = height_ratios) #, gridspec_kw={'height_ratios': [8, 4, 4]})

# for i in range(len(X0_test)):
#     Xtest_i = model_ps.simulate(X0_test[i], T).T

#     functions.plotPhasePlane(axs[0], fmodel(fI), xmesh, ymesh, Xtest_ref[i])
#     functions.plotTimeSeries(axs[1:n+1], Xtest_ref[i], T) #, plot_traj=2)
#     functions.plotTimeSeries(axs[1:n+1], Xtest_i, T, plot_traj=3)
#     # functions.plotTimeSeries(axs[n+1], Idata, T, Xlims = [-0.1*Iext, 1.1*Iext])

# fig.savefig(savepath+"/verification-FHN-A-1"+savelabel+".png", bbox_inches="tight")
# plt.close()






# # Xtraj_comp = model_ps.simulate(X0[1], T)
# # fig, axs = plt.subplots(n, 1, sharex=True, figsize=(7, 9))
# # for i in range(n):
# #     axs[i].plot(T, Xtraj_comp[:, i], 'k', label='numerical derivative')
# # fig.savefig(savepath+"/prediction_X0="+str(X0[i]).replace(".","")+"-FHN-A-1.png", bbox_inches="tight")
# # plt.close()



