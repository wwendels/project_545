# models for analysis 2D ODE systems
# Mike Wendels, last modified 2/19/2023

import numpy as np

# persistent sodium plus potassium (I_{Na,p} + I_K) model
class pNK:
    
    def __init__(self,fIext,C,gL,gNa,gK,EL,ENa,EK,minf,ninf,tau,dminf,dninf,dtau):
        self.fIext = fIext
        self.C = C
        self.gL = gL
        self.gNa = gNa
        self.gK = gK
        self.EL = EL
        self.ENa = ENa
        self.EK = EK
        self.minf = minf
        self.ninf = ninf
        self.tau = tau
        self.dminf = dminf
        self.dninf = dninf
        self.dtau = dtau

    def f1(self, x1: float, x2: float, t: float) -> np.array:
        return (self.fIext(t) - self.gL*(x1-self.EL) - self.gNa*self.minf(x1)*(x1-self.ENa) - self.gK*x2*(x1-self.EK))/self.C

    def f2(self, x1: float, x2: float, t: float) -> np.array:
        return (self.ninf(x1)-x2)*self.tau(x1)

    def f(self, x: np.array, t: float) -> np.array:
        IL = self.gL*(x[0]-self.EL)
        INa = self.gNa*self.minf(x[0])*(x[0]-self.ENa)
        IK = self.gK*x[1]*(x[0]-self.EK)
        x1dot = (self.fIext(t) - IL - INa - IK)/self.C
        x2dot = (self.ninf(x[0])-x[1])*self.tau(x[0])
        return np.array([x1dot,x2dot])
        # return np.array([(self.fIext(t) - self.gL*(x[0]-self.EL) - self.gNa*self.minf(x[0])*(x[0]-self.ENa) - self.gK*x[1]*(x[0]-self.EK))/self.C, (self.ninf(x[0])-x[1])*self.tau(x[0])])

    def f0(self, x: np.array, t: float, Iext: int = 0) -> np.array:
        # f but with Iext a constant value (default 0), for plotting reference nullclines
        # separate function from f for efficiency in timestepping with f
        return np.array([(Iext-self.gL*(x[0]-self.EL) - self.gNa*self.minf(x[0])*(x[0]-self.ENa) - self.gK*x[1]*(x[0]-self.EK))/self.C, (self.ninf(x[0])-x[1])*self.tau(x[0])])

    # x = V, y = n
    def df1_dx1(self, x: np.array) -> np.array:
        return -self.gL - self.gNa*(self.dminf(x[0])*(x[0]-self.ENa) + self.minf(x[0])) - self.gK*x[1]

    def df1_dx2(self, x: np.array) -> np.array:
        return -self.gK*(x[0]-self.EK)

    def df2_dx1(self, x: np.array) -> np.array:
        return (self.dninf(x[0])*self.tau(x[0]) - (self.ninf(x[0])-x[1])*self.dtau(x[0]))/(self.tau(x[0])**2)

    def df2_dx2(self, x: np.array) -> np.array:
        return -1/float(self.tau(x[0]))

    def jacobian(self, x: np.array) -> np.array:
        return np.array( [ [self.df1_dx1(x),self.df1_dx2(x)], [self.df2_dx1(x),self.df2_dx2(x)] ] )

    def timestepping(self, x0: np.array, dt: float, tend: float, method="RK4") -> np.array:

        nt = int(tend/dt)
        x = np.zeros((nt+1,2))
        x[0,:] = x0

        if method == "FE":
            for i in range(1,nt+1):
                t = i*dt
                x[i,:] = x[i-1,:] + dt*self.f(x[i-1,:],t)

        else: # RK4
            for i in range(1,nt+1):
                t = i*dt

                x1 = x[i-1,:]
                f1 = self.f(x1,t)
                x2 = x[i-1,:] + dt*f1/2.
                f2 = self.f(x2,t+dt/2.)
                x3 = x[i-1,:] + dt*f2/2.
                f3 = self.f(x3,t+dt/2.)
                x4 = x[i-1,:] + dt*f3
                f4 = self.f(x4,t+dt)
                x[i,:] = x[i-1,:] + dt*(f1+f2+f3+f4)/6.

        return x

# FitzHugh-Nagumo model
class FHN:
    
    def __init__(self,fIext,a,b,c):
        self.fIext = fIext
        self.a = a
        self.b = b
        self.c = c

    def f1(self, x1: float, x2: float, t: float) -> np.array:
        return x1*(self.a-x1)*(x1-1) - x2 + self.fIext(t)

    def f2(self, x1: float, x2: float, t: float) -> np.array:
        return self.b*x1 - self.c*x2

    def f(self, x: np.array, t: float) -> np.array:        
        x1dot = x[0]*(self.a-x[0])*(x[0]-1) - x[1] + self.fIext(t)
        x2dot = self.b*x[0] - self.c*x[1]
        return np.array([x1dot,x2dot])

    def f0(self, x: np.array, t: float, Iext: int = 0) -> np.array:
        # f but with Iext a constant value (default 0), for plotting reference nullclines
        # separate function from f for efficiency in timestepping with f
        return np.array([x[0]*(self.a-x[0])*(x[0]-1) - x[1] + Iext, self.b*x[0] - self.c*x[1]])

    def df1_dx1(self, x: np.array) -> np.array:
        return (self.a-x[0])*(2*x[0]-1)-x[0]*(x[0]-1)

    def df1_dx2(self, x: np.array) -> np.array:
        return -1

    def df2_dx1(self, x: np.array) -> np.array:
        return self.b

    def df2_dx2(self, x: np.array) -> np.array:
        return -self.c

    def jacobian(self, x: np.array) -> np.array:
        return np.array( [ [self.df1_dx1(x),self.df1_dx2(x)], [self.df2_dx1(x),self.df2_dx2(x)] ] )

    def timestepping(self, x0: np.array, dt: float, tend: float, method="RK4") -> np.array:

        nt = int(tend/dt)
        x = np.zeros((nt+1,2))
        x[0,:] = x0

        if method == "FE":
            for i in range(1,nt+1):
                t = i*dt
                x[i,:] = x[i-1,:] + dt*self.f(x[i-1,:],t)

        else: # RK4
            for i in range(1,nt+1):
                t = i*dt

                x1 = x[i-1,:]
                f1 = self.f(x1,t)
                x2 = x[i-1,:] + dt*f1/2.
                f2 = self.f(x2,t+dt/2.)
                x3 = x[i-1,:] + dt*f2/2.
                f3 = self.f(x3,t+dt/2.)
                x4 = x[i-1,:] + dt*f3
                f4 = self.f(x4,t+dt)
                x[i,:] = x[i-1,:] + dt*(f1+f2+f3+f4)/6.

        return x

# Izhikevich model
class Izh:

    def __init__(self,fIext,C,k,vr,vt,vpeak,a,b,c,d):
        self.fIext = fIext
        self.C = C
        self.k = k
        self.vr = vr
        self.vt = vt
        self.vpeak = vpeak
        self.a = a
        self.b = b
        self.c = c
        self.d = d

    def f1(self, x1: float, x2: float, t: float) -> np.array:
        return (self.fIext(t) + self.k*(x1-self.vr)*(x1-self.vt) - x2)/self.C

    def f2(self, x1: float, x2: float, t: float) -> np.array:
        return self.a*(self.b*(x1-self.vr)-x2)

    def f(self, x: np.array, t: float) -> np.array:
        x1dot = (self.fIext(t) + self.k*(x[0]-self.vr)*(x[0]-self.vt) - x[1])/self.C
        x2dot = self.a*(self.b*(x[0]-self.vr)-x[1])
        return np.array([x1dot,x2dot])

    def f0(self, x: np.array, t: float, Iext: int = 0) -> np.array:
        # f but with Iext a constant value (default 0), for plotting reference nullclines
        # separate function from f for efficiency in timestepping with f
        x1dot = (Iext + self.k*(x[0]-self.vr)*(x[0]-self.vt) - x[1])/self.C
        x2dot = self.a*(self.b*(x[0]-self.vr)-x[1])
        return np.array([x1dot,x2dot])        

    def df1_dx1(self, x: np.array, t: float) -> float:
        # return (2*self.k*x[0]-self.vr-self.vt)/self.C
        return self.k*x[0]*((x[0]-self.vt) + (x[0]-self.vr))/self.C

    def df1_dx2(self, x: np.array, t: float) -> float:
        return -1.0/self.C

    def df2_dx1(self, x: np.array, t: float) -> float:
        return self.a*self.b

    def df2_dx2(self, x: np.array, t: float) -> np.array:
        return -self.a

    def jacobian(self, x: np.array, t: float) -> np.array:
        return np.array( [ [self.df1_dx1(x, t), self.df1_dx2(x, t)], [self.df2_dx1(x, t), self.df2_dx2(x, t)] ] )

    def timestepping(self, x0: np.array, dt: float, tend: float, method="RK4") -> np.array:

        nt = int(tend/dt)
        x = np.zeros((nt+1,2))
        x[0,:] = x0

        if method == "FE":
            for i in range(1,nt+1):
                t = i*dt
                x[i,:] = x[i-1,:] + dt*self.f(x[i-1,:],t)

                if x[i,0] >= self.vpeak:
                    x[i-1,0] = self.vpeak
                    x[i,0] = self.c
                    x[i,1] += self.d

        else: # RK4
            for i in range(1,nt+1):
                t = i*dt

                x1 = x[i-1,:]
                f1 = self.f(x1,t)
                x2 = x[i-1,:] + dt*f1/2.
                f2 = self.f(x2,t+dt/2.)
                x3 = x[i-1,:] + dt*f2/2.
                f3 = self.f(x3,t+dt/2.)
                x4 = x[i-1,:] + dt*f3
                f4 = self.f(x4,t+dt)
                x[i,:] = x[i-1,:] + dt*(f1+f2+f3+f4)/6.

                if x[i,0] >= self.vpeak:
                    x[i-1,0] = self.vpeak
                    x[i,0] = self.c
                    x[i,1] += self.d

        return x

# CURRENT FUNCTIONS

def fConstant(t,I):
    # constant current
    return I

def fHeaviside(t,I,tthres):
    # jump to 0 after tthres
    if t<=tthres:
        return I
    else:
        return 0

def fPiecewiseLinear(t,X,Y):
    # connects points (x_i,y_i) linearly, where x_i is the ith element of X and y_i the ith element of Y

    nx = len(X)
    i = 0
    if t <= X[0]:
        i = -1

    if t >= X[-1]:
        i = nx-1

    while t >= X[i] and i < nx-1:
        i += 1
    i -= 1

    if i == -1: # (possible) extrapolation to the left (if t < X[0])
        return Y[1] + (t - X[1])*(Y[0] - Y[1])/(X[0] - X[1])

    elif i < nx-1:
        return Y[i] + (t - X[i])*(Y[i+1] - Y[i])/(X[i+1] - X[i])

    elif i == nx-1: # (possible) extrapolation to the right
        return Y[-2] + (t - X[-2])*(Y[-1] - Y[-2])/(X[-1] - X[-2])

def fPulses(t,Iref,Pulses):
    # multiple pulses, between times t_{i,1} and t_{i,2} with T=[ [t_{1,1},t_{1,2}], [t_{2,1},t_{2,2}], ... ] return I1 else I2
    val = Iref
    for P in Pulses:
        if t>=P[0] and t<=P[1]:
            val = P[2]
    return val