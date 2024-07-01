# -*- coding: utf-8 -*-
"""
Created on Sun Jun  9 13:48:26 2024

@author: zaidh
"""
import numpy as np
import matplotlib.pyplot as plt
import time
import math
from scipy.integrate import odeint

E = 6
delta = 1
x0 = 10
g = 1
#EDOS SIMPLIFICATION
###############################################################################
SD = 0.15 #Where I want the  r to start
n0 = 10 #Where I want the V to start
# Initial conditions.
dt = 0.001
ti = 0
Tmax= 2

t0 = 10
t2 = 30
I = 0

tvec=np.arange(ti,Tmax,dt)
vri = n0
rri = SD
vrri = [vri, rri]

def I0(t,ti=t0,tf=t2,Ia=I):
    if t > ti and t < tf:
        return Ia
    else:
        return 0  #We define the applied current to the network

def f_vrr(vrr, t,x=x0,gamma=delta,E=E,g=g):
    vr, rr = vrr #initial conditions
    dvrdt = (vr**2) + x + g*rr*(E-vr) -(np.pi**2)*rr**2 +I0(t) #vr ODE
    drrdt = gamma/np.pi + 2*rr*vr -g*rr**2 #rr ODE
    return [dvrdt, drrdt]

vrr = odeint(f_vrr, vrri, tvec)

# We define the X and Y vectors from the result.
vr = vrr[:, 0]
rr = vrr[:, 1]

########################################

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