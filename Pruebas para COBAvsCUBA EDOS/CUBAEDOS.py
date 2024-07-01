# -*- coding: utf-8 -*-
"""
Created on Sun Jun  9 16:32:07 2024

@author: zaidh
"""
import numpy as np
import matplotlib.pyplot as plt
import time
import math
from scipy.integrate import odeint

#%%
#EDOS SIMPLIFICATION
###############################################################################

# Initial conditions.
dt = 0.001
ti = 0
Tmax= 2

t0 = 10
t2 = 40
I = 0

coupling = 50
delta = 1
x0 = 0
SD = 0.001 #Where I want the Fr to start
n0 = -2 #Where I want the V to start

tvec=np.arange(ti,Tmax,dt)
vri = n0
rri = SD
vrri = [vri, rri]

def I0(t,ti=t0,tf=t2,Ia=I):
    if t > ti and t < tf:
        return Ia
    else:
        return 0  #We define the applied current to the network


def f_vrr(vrr, t,x=x0,J=coupling,gamma=delta):
    vr, rr = vrr #initial conditions
    dvrdt = (vr**2) + x + J*rr + I0(t) -(np.pi**2)*rr**2 #vr ODE
    drrdt = gamma/np.pi + 2*rr*vr #rr ODE
    return [dvrdt, drrdt]

vrr = odeint(f_vrr, vrri, tvec)

# We define the X and Y vectors from the result.
vr = vrr[:, 0]
rr = vrr[:, 1]

#%%


plt.plot(tvec,rr)
plt.title('Firing Rates',size=25)
plt.xlabel('Time',size=15)
plt.ylabel('$r$',size=15)
plt.legend()
plt.show()  

plt.plot(tvec,vr)
plt.title('Voltage',size=25)
plt.xlabel('Time',size=15)
plt.ylabel('$v$',size=15)
plt.legend()
plt.show()