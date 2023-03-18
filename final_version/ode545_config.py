# cases (ODE systems with parameters) to simulate with the main function
# Mike Wendels, last modified 3/17/2023

import numpy as np
import ode545_models as model

def getModel(case):

    if case == "FHN-A":

        xmesh = -0.5, 1.0, 30, 50, "$V$"
        ymesh = -0.05, 0.3, 30, 50, "$w$"
        tmesh = 500.,0.01
        X0 = [np.array([0,0.15])]
        eqtol = 1e-3

        a = -0.1
        b = 0.01
        c = 0.02


    elif case == "FHN-B":

        xmesh = -0.5, 1.0, 30, 50, "$V$"
        ymesh = -0.05, 0.3, 30, 50, "$w$"
        tmesh = 500.,0.01
        X0 = [np.array([0, 0.15])]
        eqtol = 1e-3

        a = 0.1
        b = 0.01
        c = 0.02


    elif case == "FHN-C":

        xmesh = -0.5, 1.0, 30, 50, "$V$"
        ymesh = -0.05, 0.2, 30, 50, "$w$"
        tmesh = 500.,0.01
        X0 = [np.array([0,0.5])]
        eqtol = 1e-3

        a = 0.1
        b = 0.01
        c = 0.1

    elif case == "PNK-A":

        xmesh = -90, 30.0, 30, 50, "$V$"
        ymesh = 0.0, 1.0, 30, 50, "$n$"
        tmesh = 500.,0.01
        X0 = [np.array([0,0.5])]
        eqtol = 1e-1

        f_minf = lambda V,Vhalf,k : 1/(1+np.exp((Vhalf-V)/k)) # np.power(1+np.exp((Vhalf-V)/k),-1)
        df_minf = lambda V,Vhalf,k : np.exp((Vhalf-V)/k)/(1+np.exp((Vhalf-V)/k))/k

        C = 1
        gL = 8
        gNa = 20
        gK = 10
        EL = -80
        ENa = 60
        EK = -90
        minf = lambda V : f_minf(V,-20,15)
        ninf = lambda V : f_minf(V,-25,5)
        tau = lambda V : 1
        dminf = lambda V : df_minf(V,-20,15)
        dninf = lambda V : df_minf(V,-25,5)
        dtau = lambda V : 0

    elif case == "PNK-B":

        xmesh = -90, 30.0, 30, 50, "$V$"
        ymesh = 0.0, 1.0, 30, 50, "$n$"
        tmesh = 500.,0.01
        X0 = [np.array([0,0.5])]
        eqtol = 1e-1

        f_minf = lambda V,Vhalf,k : 1/(1+np.exp((Vhalf-V)/k)) # np.power(1+np.exp((Vhalf-V)/k),-1)
        df_minf = lambda V,Vhalf,k : np.exp((Vhalf-V)/k)/(1+np.exp((Vhalf-V)/k))/k

        C = 1
        gL = 8
        gNa = 20
        gK = 10
        EL = -80
        ENa = 60
        EK = -90
        minf = lambda V : f_minf(V,-20,15)
        ninf = lambda V : f_minf(V,-45,5)
        tau = lambda V : 1
        dminf = lambda V : df_minf(V,-20,15)
        dninf = lambda V : df_minf(V,-45,5)
        dtau = lambda V : 0

    else:
        xmesh = None
        ymesh = None
        tmesh = None
        X0 = None
        eqtol = None

    if case[:3] == "FHN":
        fmodel = lambda f : model.FHN(fIext=f,a=a,b=b,c=c)
    elif case[:3] == "PNK":
        fmodel = lambda f : model.pNK(fIext=f,C=C,gL=gL,gNa=gNa,gK=gK,EL=EL,ENa=ENa,EK=EK,minf=minf,ninf=ninf,tau=tau,dminf=dminf,dninf=dninf,dtau=dtau)
    else:
        fmodel = None
        print("ERROR: specify model")

    return fmodel,xmesh,ymesh,tmesh,X0,eqtol
