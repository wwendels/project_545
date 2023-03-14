# cases (2D ODE systems with parameters) to simulate with the main function
# Mike Wendels, last modified 2/19/2023

import numpy as np
import ode2d_models as model

def getModel(case):

    if case == "IZH-A":

        xmesh = -100.0, 50.0, 30, 500, "$v$"
        ymesh = -50.0, 200.0, 30, 500, "$u$"
        tmesh = 500.,0.01 #200.,0.1
        X0 = [np.array([-45,-20])] #[np.array([-60,0]),np.array([-45,-20])]
        eqtol = 1e-3
        model_lab = "IZH"

        C = 100
        k = 0.7
        vr = -60
        vt = -40
        vpeak = 35
        a = 0.03
        b = -2
        c = -50
        d = 100

        # Iext = [0,20,40,60,80,100,200,300]


    elif case == "IZH-B":
        print("HERRREEEEEE")

        xmesh = -100.0, 50.0, 30, 500, "$v$"
        ymesh = -50.0, 250.0, 30, 500, "$u$"
        xmesh = -60.0, -40.0, 30, 500, "$v$"
        ymesh = 40.0, 100.0, 30, 500, "$u$"
        xmesh = -100.0, 50.0, 30, 500, "$v$"
        ymesh = -50.0, 300.0, 30, 500, "$u$"
        tmesh = 500.,0.01 #200.,0.1
        X0 = [np.array([-60,100]), np.array([-70,100])] #[np.array([-40,50])] #[np.array([-60,100])] # compared to (-60,0) in model A
        eqtol = 1e-3
        model_lab = "IZH"

        C = 100
        k = 0.7
        vr = -60
        vt = -40
        vpeak = 35
        a = 0.03
        b = 5 # compared to -2 in model A
        c = -50
        d = 100

        # Iext = [200,300,400,500] #[120,124.5,125,126.5,127,127.5]


    elif case == "IZH-C":

        xmesh = -100.0, 50.0, 30, 500, "$v$"
        ymesh = -50.0, 450.0, 30, 500, "$u$"
        tmesh = 500.,0.01 #200.,0.1
        X0 = [np.array([-80,0])]
        eqtol = 1e-3
        model_lab = "IZH"

        C = 150
        k = 1.2
        vr = -75
        vt = -45
        vpeak = 50
        a = 0.01
        b = 5
        c = -56
        d = 130

        # Iext = [0,300,370,500,550]


    elif case == "IZH-D":

        xmesh = -100.0, 50.0, 30, 500, "$v$"
        ymesh = 0,20000,30,500,"$u$" 
        ymesh = -50.0, 250.0, 30, 500, "$u$"
        tmesh = 500.,0.01
        X0 = [np.array([30,19000])] 
        X0 = [np.array([-50,50])] #[np.array([-70,200])] #[np.array([-60,0])]
        eqtol = 1e-3
        model_lab = "IZH"

        C = 50
        k = 0.5
        vr = -60
        vt = -45
        vpeak = 40
        a = 0.02
        b = 0.5
        c = -35 #fig 8.35
        d = 60

        # Iext = [0,10,30,50,70]


    elif case == "IZH-E":

        xmesh = -100.0, 50.0, 30, 500, "$v$"
        ymesh = -50.0, 200.0, 30, 500, "$u$"
        tmesh = 500.,0.01 #200.,0.1
        X0 = [np.array([-40,150])] #[np.array([-60,0]),np.array([-45,-20])]
        eqtol = 1e-3
        model_lab = "IZH"

        C = 100
        k = 0.7
        vr = -60
        vt = -40
        vpeak = 35
        a = 0.5
        b = 0.2
        c = -65
        d = 2

        # Iext = [0,20,40,60,80,100,200,300]


    elif case == "FHN-A":

        xmesh = -0.5, 1.0, 30, 50, "$V$"
        ymesh = -0.05, 0.2, 30, 50, "$w$"
        tmesh = 500.,0.01
        X0 = [np.array([0,0.15])] #, np.array([0,0.2])]
        eqtol = 1e-3
        model_lab = "FHN"

        a = 0.1
        b = 0.01
        c = 0.02

        # Iext = [0,10,30,50,70]


    elif case == "FHN-B":

        xmesh = -0.5, 1.0, 30, 50, "$V$"
        ymesh = -0.05, 0.2, 30, 50, "$w$"
        tmesh = 500.,0.01
        X0 = [np.array([0, 0.15]), np.array([0, 0.2])]
        eqtol = 1e-3
        model_lab = "FHN"

        a = -0.1
        b = 0.01
        c = 0.02

        # Iext = [0,10,30,50,70]


    elif case == "FHN-C":

        xmesh = -0.5, 1.0, 30, 50, "$V$"
        ymesh = -0.05, 0.2, 30, 50, "$w$"
        tmesh = 500.,0.01
        X0 = [np.array([0,0.5])]
        eqtol = 1e-3
        model_lab = "FHN"

        # a = -0.1 #
        a = 0.1
        b = 0.01
        c = 0.1

        # Iext = [0,10,30,50,70]


    elif case == "PNK-A":

        xmesh = -90, 30.0, 30, 50, "$V$"
        ymesh = 0.0, 1.0, 30, 50, "$n$"
        tmesh = 500.,0.01
        X0 = [np.array([0,0.5])]
        eqtol = 1e-1
        model_lab = "PNK"

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

        # Iext = [0,10,30,50,70]


    elif case == "PNK-B":

        xmesh = -90, 30.0, 30, 50, "$V$"
        ymesh = 0.0, 1.0, 30, 50, "$n$"
        tmesh = 500.,0.01
        X0 = [np.array([0,0.5])]
        eqtol = 1e-1
        model_lab = "PNK"

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

        # Iext = [0,10,30,50,70]

    else:
        # model_lab = None
        xmesh = None
        ymesh = None
        tmesh = None
        X0 = None
        eqtol = None

    if case[:3] == "IZH":
        fmodel = lambda f : model.Izh(fIext=f,C=C,k=k,vr=vr,vt=vt,vpeak=vpeak,a=a,b=b,c=c,d=d)
    elif case[:3] == "FHN":
        fmodel = lambda f : model.FHN(fIext=f,a=a,b=b,c=c)
    elif case[:3] == "PNK":
        fmodel = lambda f : model.pNK(fIext=f,C=C,gL=gL,gNa=gNa,gK=gK,EL=EL,ENa=ENa,EK=EK,minf=minf,ninf=ninf,tau=tau,dminf=dminf,dninf=dninf,dtau=dtau)
    else:
        fmodel = None
        print("ERROR: specify model")

    return fmodel,xmesh,ymesh,tmesh,X0,eqtol #Iext,eqtol
