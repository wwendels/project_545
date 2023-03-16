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
case = "FHN-B"
savepath_main = cwd+"/FHN_model_B/"

fmodel, xmesh, ymesh, tmesh, X0, eqtol = config.getModel(case)  # gets (detailed) input regarding model, mesh, initial value and tolerance for finding equilibria from ode2d_config
tmesh = 1000., 0.005 #5 #1                                             # time mesh (different subfolders will be made for different time meshes)
seed = 38 #37

if case[:3] == "FHN":
    Xlabs = ["$V$","$w$"]
else:
    print("ERROR: specify Xlabs")

# ----- parameters input current function fI ----- #
#Iext = 0.04 #100                                                      # input current
# tthres = 500                                                     # threshold current
#Pulses = [[180,200,Iext],[580,600,Iext]]                        # time intervals for current pulses 
# Iref = 0                                                        # reference current
# fI_lab = 1                                                      # choice for which current function (see output section for which label corresponds to which function)

# ----- SINDy training parameters ----- #
# NOTE: applying SINDy only possible if model_recovery == True and fI_lab == 1 
# model_recovery = True   # to apply SINDy

# general training parameters
train_opt = 6           # specifies X0 and I for training data (see below)
test_opt = 1            # specifies X0 and I for test data (see below)
eta = 0 #.001                 # std Gaussian noise applied

# SINDy specific training parameters
poly_order = 3          # highest polynomial degree
threshold = 0.005 #0.1       # STLSQ optimizer threshold (should be >= smallest coefficient in model)

burnin = 0.1

# if model_recovery:
savedir = f"SINDy_" + case + f"_eta={eta}_poly-order={poly_order}_thr={threshold}_seed={seed}_T={tmesh[0]}_dt={tmesh[1]}_burnin={burnin}"  # name subfolder created for current run
savelabel = f"train-opt={train_opt}_test-opt={test_opt}"                            # string with which each file name in the subfolder starts
# else:
#     savedir = f"f{fI_lab}_tend={tmesh[0]}_dt={tmesh[1]}"            # name subfolder created for current run
#     savelabel = f"Iext={Iext}"                                      # string with which each file name in the subfolder starts


# ===== OUTPUT ===== #

np.random.seed(seed)

savepath = savepath_main+savedir

if not os.path.exists(savepath_main):
    os.mkdir(savepath_main)

if not os.path.exists(savepath):
    os.mkdir(savepath)

# ----- current functions ----- #

# if fI_lab == 1: # constant current
#     fI = lambda t : model.fConstant(t,Iext)
# elif fI_lab == 2: # Heaviside function
#     fI = lambda t : model.fHeaviside(t,Iext,tthres)
# elif fI_lab == 3: # linear interpolation between given points
#     fI = lambda t : model.fPiecewiseLinear(t,[0,tthres,tmesh[0]],[0,Iext,Iext])
# elif fI_lab == 4: # pulses of stength Pulses[2] for given time intervals (Pulses[0],Pulses[1])
#     fI = lambda t : model.fPulses(t,Iref,Pulses)
# else:
#     print("ERROR [ode2d_main]: specify valid option current function fI_lab")
#     exit(1)

# ----- PySINDy training/test data ----- #

if train_opt == 0:
    X0_train = [np.array([0, 0.15])]
    I_train = [0]
if train_opt == 1:
    X0_train = [np.array([0, 0.2])]
    I_train = [0]
elif train_opt == 2:
    X0_train = [np.array([0, 0.2]), np.array([0.4, 0]), np.array([0.8, 0.1])]
    I_train = [0]
elif train_opt == 3:
    X0_train = [np.array([0, 0.2])]
    I_train = [0, 0.01, 0.02] #, 0.03]
elif train_opt == 4:
    X0_train = [np.array([0, 0.2])]
    I_train = [0.04, 0.05, 0.06] #[0, 0.02, 0.04, 0.06]
elif train_opt == 5:
    X0_train = [np.array([0, 0.2]), np.array([0.4, 0]), np.array([0.8, 0.1])]
    I_train = [0, 0.01, 0.02] #, 0.03]
elif train_opt == 6:
    X0_train = [np.array([0, 0.2]), np.array([0.4, 0]), np.array([0.8, 0.1])]
    I_train = [0, 0.02, 0.04, 0.06]
elif train_opt == 7:
    X0_train = [np.array([0, 0]), np.array([0, 0.1]), np.array([0, 0.2]), 
                np.array([0.4, 0]), np.array([0.4, 0.1]), np.array([0.4, 0.2]),
                np.array([0.8, 0]), np.array([0.8, 0.1]), np.array([0.8, 0.2])]
    I_train = [0]
elif train_opt == 8:
    X0_train = [np.array([0, 0]), np.array([0, 0.05]), np.array([0, 0.1]), np.array([0, 0.15]), np.array([0, 0.2]), 
                np.array([0.2, 0]), np.array([0.2, 0.05]), np.array([0.2, 0.1]), np.array([0.2, 0.15]), np.array([0.2, 0.2]),
                np.array([0.4, 0]), np.array([0.4, 0.05]), np.array([0.4, 0.1]), np.array([0.4, 0.15]), np.array([0.4, 0.2]),
                np.array([0.6, 0]), np.array([0.6, 0.05]), np.array([0.6, 0.1]), np.array([0.6, 0.15]), np.array([0.6, 0.2]),
                np.array([0.8, 0]), np.array([0.8, 0.05]), np.array([0.8, 0.1]), np.array([0.8, 0.15]), np.array([0.8, 0.2])]
    I_train = [0]
elif train_opt == 9:
    X0_train = [np.array([0, 0]), np.array([0, 0.1]), np.array([0, 0.2]), 
                np.array([0.4, 0]), np.array([0.4, 0.1]), np.array([0.4, 0.2]),
                np.array([0.8, 0]), np.array([0.8, 0.1]), np.array([0.8, 0.2])]
    I_train = [0]



if test_opt == 0:
    X0_test = [np.array([0, 0.1]), np.array([0.4, 0.2]), np.array([0.8, 0])]
    I_test = [0]
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

# # trajectories state of system
# Xtraj, T = functions.trajectories(fmodel(fI), X0_train, tmesh, method="RK4") # list of arrays of ndims x nt ( see print(Xtraj[0].shape) )

# # initialization summary plot
# n, nt = Xtraj[0].shape                              # n: dimensionality of the ODE, nt: number of time steps ( = len(T) )
# dt = T[1] - T[0]                                    # Delta t

# # trajectory input current
# Idata = np.zeros((1,nt))
# for i in range(nt):
#     Idata[0,i] = fI(T[i])


# plot phase plane, time series state of system and time series input current
for i in range(len(X0_train)):
    for j in range(len(I_train)):
        # fI = functionInputCurrent(fI_lab, Iext, tthres, tmesh, Iref, Pulses)
        fI = lambda t : model.fConstant(t,I_train[j])

        # trajectories state of system
        Xtraj, T = functions.trajectories2(fmodel(fI), X0_train[i], tmesh, method="RK4") # list of arrays of ndims x nt ( see print(Xtraj[0].shape) )

        # initialization summary plot
        n, nt = Xtraj.shape                              # n: dimensionality of the ODE, nt: number of time steps ( = len(T) )
        # print(n, nt)
        dt = T[1] - T[0]                                    # Delta t

        # trajectory input current
        Idata = np.zeros((1,nt))
        for k in range(nt):
            Idata[0,k] = fI(T[k])

        height_ratios = [2]+[1 for i in range(n)]
        fig, axs = plt.subplots(n+1, 1, figsize=(12, 12), height_ratios = height_ratios) #, gridspec_kw={'height_ratios': [8, 4, 4]})

        functions.plotPhasePlane(axs[0], fmodel(fI), xmesh, ymesh, Xtraj) #[i])
        functions.plotTimeSeries(axs[1:n+1], Xtraj, T, Xlabs=Xlabs) #[i], T)
        # functions.plotTimeSeries(axs[n+1], Idata, T, Xlims = [-0.1*I_train[j], 1.1*I_train[j]])

        fig.savefig(savepath+"/analysis_X0="+str(X0_train[i]).replace("0. ","0").replace(" ",",")+"_I="+str(I_train[j])+".png", bbox_inches="tight")
        plt.close()

####################
# PYSINDY RECOVERY #
####################
# extracts the equations for the dynamical system according to synthetic data generated from it, possibly with noise added

# ----- model training ----- #

# training
X_train = np.array([], dtype=np.int64).reshape(0,n+1)

for X0i in X0_train:
    for Ii in I_train:
        fIi = lambda t : model.fConstant(t,Ii)
        X_train_i, T = functions.trajectories2(fmodel(fIi), X0i, tmesh, method="RK4")
        X_train_i += np.random.normal(loc=0, scale=eta, size=X_train_i.shape) #= functions.addWhiteNoise(X_train_i, eta)
        I_append = np.array([ Ii for k in range(len(X_train_i.T))]).reshape(len(X_train_i.T),1)
        # nstart = int(np.ceil(burnin*float(len(X_train_i))))
        # X_train_i_filtered = X_train_i[nstart:]
        data_train_i = np.hstack([X_train_i.T, I_append])

        nstart = int(np.ceil(burnin*float(len(data_train_i))))
        data_train_i_filtered = data_train_i[nstart:]
        print(data_train_i_filtered.shape)

        X_train = np.vstack([X_train, data_train_i_filtered])

model_ps = ps.SINDy(optimizer=ps.STLSQ(threshold=threshold), feature_library=ps.PolynomialLibrary(degree=poly_order))
model_ps.fit(X_train, t=dt)

# save model in a text file (and print on the command line)
# NOTE: model_ps.print() does the same but cannot be used to obtain the string for the text file as it only prints and does not return a string
model_txt = open(savepath+"/model_"+savelabel+".txt","w")
eqns = model_ps.equations(precision=3)
feature_names = model_ps.feature_names
for i, eqn in enumerate(eqns):
    names = "(" + feature_names[i] + ")"
    print(names + "' = " + eqn)
    model_txt.write(names + " = " + eqn +" \n")


# ----- plot of reference cases (true vs. approximated) for test data ----- #
fig, axs = plt.subplots(n, 1, figsize=(12, 12)) 

error_test = 0
for X0i in X0_test:
    for Ii in I_test:

        # get data
        fIi = lambda t : model.fConstant(t,Ii)
        X_test_i_ref, T = functions.trajectories2(fmodel(fIi), X0i, tmesh, method="RK4")
        X_test_i_sim = model_ps.simulate(np.append(X0i,Ii), T).T # NOTE: if no I dependence, there will be (seemingly) less plots!

        # plot data
        functions.plotTimeSeries(axs, X_test_i_sim[:-1], T, plot_traj=2)
        functions.plotTimeSeries(axs, X_test_i_ref, T, plot_traj=3, Xlabs=Xlabs)

        # update error
        error_test += np.power(X_test_i_sim[:-1] - X_test_i_ref,2)

error_test = np.power(np.mean(error_test),0.5)
model_txt.write("error test = " + str(error_test.round(3)) +" \n")

fig.savefig(savepath+"/model_verification_test_"+savelabel+".png", bbox_inches="tight")
plt.close()


# ----- plot of reference cases (true vs. approximated) for training data ----- #
fig, axs = plt.subplots(n, 1, figsize=(12, 12))

error_train = 0
for X0i in X0_train:
    for Ii in I_train:

        # get data
        fIi = lambda t : model.fConstant(t,Ii)
        X_train_i_ref, T = functions.trajectories2(fmodel(fIi), X0i, tmesh, method="RK4") 
        X_train_i_ref += np.random.normal(loc=0, scale=eta, size=X_train_i.shape)
        X_train_i_sim = model_ps.simulate(np.append(X0i,Ii), T).T

        # plot data
        functions.plotTimeSeries(axs, X_train_i_sim[:-1], T, plot_traj=2)
        functions.plotTimeSeries(axs, X_train_i_ref, T, plot_traj=3, Xlabs=Xlabs)

        # update error
        error_train += np.power(X_train_i_sim[:-1] - X_train_i_ref,2)

error_train = np.power(np.mean(error_train),0.5)
model_txt.write("error train = " + str(error_train) +" \n")

fig.savefig(savepath+"/model_verification_train_"+savelabel+".png", bbox_inches="tight")
plt.close()

model_txt.close()

