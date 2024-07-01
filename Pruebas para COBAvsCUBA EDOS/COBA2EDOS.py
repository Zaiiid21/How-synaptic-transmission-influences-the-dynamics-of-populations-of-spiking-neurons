# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 00:21:02 2024

@author: zaidh
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import math
from scipy.integrate import odeint

#VARIABLES
###############################################################################
ti = 0
Tmax= 50
dt=0.001

t0 = 10
t2 = 30
Iexc = 5
Iinh = 5

g = 1
Ee = 30
Ei = -30
delta = 1
x0e = 5
x0i = -5
SDe = 0 #Where I want the Fre to start
n0e = -1 #Where I want the Ve to start
SDi = 0 #Where I want the Fri to start
n0i = -1 #Where I want the Vi to start

#EDOS SIMPLIFICATION
###############################################################################

# Initial conditions.
tvec=np.arange(ti,Tmax,dt)
vrie = n0e
rrie = SDe
vrii = n0i
rrii = SDi
vrri = [vrie, rrie, vrii, rrii]

def I0(t, Iae,ti=t0,tf=t2):
    if t > ti and t < tf:
        return Iae #Ia*np.sin((np.pi/20)*t)
    else:
        return 0  #We define the applied current to the network


def f_vrr(vrr, t,xi=x0i,xe=x0e,g=g,gamma=delta,Ee=Ee,Ei=Ei):
    vre, rre, vri, rri = vrr #initial conditions
    dvredt = (vre**2) + xe + (g+I0(t,Iexc))*rri*(Ei-vre) -(np.pi**2)*rre**2 #vr ODE
    drredt = delta/np.pi + 2*rre*vre -(g+I0(t,Iexc))*rre*rri #rr ODE
    dvridt = (vri**2) + xi + (g+I0(t,Iinh))*rre*(Ee-vri) -(np.pi**2)*rri**2 
    drridt =delta/np.pi + 2*rri*vri -(g+I0(t,Iinh))*rri*rre
    return [dvredt, drredt, dvridt, drridt]

vrr = odeint(f_vrr, vrri, tvec)

# We define the X and Y vectors from the result.
vre = vrr[:, 0]
rre = vrr[:, 1]
vri = vrr[:, 2]
rri = vrr[:, 3]

plt.plot(tvec,rre)
plt.title('Firing Rates e',size=25)
plt.xlabel('Time',size=15)
plt.ylabel('$r_e$',size=15)
plt.legend()
plt.show()  

plt.plot(tvec,rri)
plt.title('Firing Rates i',size=25)
plt.xlabel('Time',size=15)
plt.ylabel('$r_i$',size=15)
plt.legend()
plt.show()  

plt.plot(tvec,vre)
plt.title('Voltage',size=25)
plt.xlabel('Time',size=15)
plt.ylabel('$v_e$',size=15)
plt.legend()
plt.show()

plt.plot(tvec,vri)
plt.title('Voltage',size=25)
plt.xlabel('Time',size=15)
plt.ylabel('$v_i$',size=15)
plt.legend()
plt.show()