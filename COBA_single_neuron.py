# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 23:56:49 2024

@author: zaidh
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import math
from scipy.integrate import odeint

#VARIABLES
###############################################################################
N = 1      # We define the global parameters
vres = 100
ti = 0
Tmax= 10
dt=0.001
ventana = 0.001 #seconds
tau = int(ventana/dt)
refractory_period = -1  # You can adjust this value as needed.
sn = 0
g = 30
E = 6
delta = 1
x0 = -20
n0 = 0 #Where I want the V to start
###############################################################################

#FUNCTIONS AND INITATION
###############################################################################

def s(t,spike):
    return spike

def f_v(v0,t,mu,g,E,spike):
    I = + mu +1*g*(E-v0)+70
    return [(v0**2) + I,I]

tvec=np.arange(ti,Tmax,dt)
v0=np.zeros((N, len(tvec)))
v0[0][0] = n0
Isyn = np.zeros((N,len(tvec)))
Isyn[0][0] = 0    

spike_times= []
[spike_times.append([]) for x in range(N)]
#Raster = np.ones((N,len(tvec)))


start_time = time.time()

mu = x0
    

spike = [0]
S = np.zeros((1,len(tvec)))
S = S[0]
v = np.zeros((1,len(tvec)))
v = v[0]


refractory_state = np.zeros(N)
###############################################################################

#%%

#POPULATION INTEGRATION
###############################################################################

for j in range(1,len(tvec)): #j represents time step.
    spike.append(0)
    for i in range(N): #i represents neuron.
        if refractory_state[i] > 0:
            v0[i][j]= -2*vres
            refractory_state[i] -= 1  # Decrease refractory state.
        else:
            v[j-1] += v0[i][j-1]/N #voltage
            v0[i][j]=v0[i][j-1]+dt*f_v(v0[i][j-1],tvec[j-1],mu,g,E,S[j-1])[0]    #We use Euler method in order to compute the ODE's
            Isyn[i][j] = f_v(v0[i][j-1],tvec[j-1],mu,g,E,S[j-1])[1]
        #Reset the phases to 0 if they are larger than the threshold
        if v0[i][j]>vres:
            v0[i][j] = v0[i][j]-vres*2
            refractory_state[i] = refractory_period #We enter refractory time.
            #Save the spiking time when the threshold is passed.
            
            spike_times[i].append(tvec[j])
            #Raster[i][j] = 0
            spike[j] += sn
            
    if j<tau: #Distributional Dirac delta integral.
        S[j] = (np.sum(spike)/j)/N
    else:
        S[j] = (np.sum(spike[-tau:])/tau)/N 
end_time = time.time()
###############################################################################

#%%

plt.figure(figsize=(21,5))
plt.plot(tvec,v0[0],'b-',label="v")
plt.plot(spike_times[0],np.ones(len(spike_times[0]))*(vres/1.5),"or")
plt.title('One neuron',fontname='Times New Roman', fontsize=38)
plt.xlabel('Time',fontname='Times New Roman', fontsize=32)
plt.ylabel('Spikes over time',fontname='Times New Roman', fontsize=32)
plt.legend()
plt.show()

plt.figure(figsize=(21,5))
plt.plot(tvec,Isyn[0],label="$I_ syn$")
plt.plot(spike_times[0],np.ones(len(spike_times[0]))*(vres/1.5),"or")
plt.title('$I_ syn$',fontname='Times New Roman', fontsize=38)
plt.xlabel('Time',fontname='Times New Roman', fontsize=32)
plt.ylabel('$Synaptic Current$',fontname='Times New Roman', fontsize=32)
plt.legend()
plt.show()



#%%
#print('Spikes:',spike_times[0])
print("Tiempo de ejecución:", end_time - start_time, "segundos")

#%%
"""

plt.figure(figsize=(400,50))
plt.plot(tvec,S)
plt.title('Firing Rates',size=25)
plt.xlabel('Time',size=15)
plt.ylabel('$r$',size=15)
plt.legend()
plt.show()

plt.figure(figsize=(400,50))
plt.plot(tvec,v)
plt.title('Voltage',size=25)
plt.xlabel('Time',size=15)
plt.ylabel('$v$',size=15)
plt.legend()
plt.show()  

"""


#%%
#EDOS SIMPLIFICATION
###############################################################################
"""
# Initial conditions.
dt = 0.001
tvec=np.arange(ti,Tmax,dt)

C = x0+g*E;
z = math.sqrt((-g**2)+4*C)
R = -2*math.atan(g/math.sqrt((-g**2)+4*C))/math.sqrt((-g**2)+4*C)
v = -g +math.tan(math.sqrt((-g**2)+4*C)*(tvec+R)/2)*math.sqrt((-g**2)+4*C)
"""

#%%

"""
plt.figure(figsize=(400,50))
plt.plot(tvec,rr)
plt.title('Firing Rates',size=25)
plt.xlabel('Time',size=15)
plt.ylabel('$r$',size=15)
plt.legend()
plt.show()  
"""


#%%
"""
import scipy.io as sio

sio.savemat('V_COBA.mat', {'V_COBA': v})
sio.savemat('FRr_COBA.mat', {'FRr_COBA': rr})
sio.savemat('Vr_COBA.mat', {'Vr_COBA': vr})
sio.savemat('S_COBA.mat', {'S_COBA': S})
#sio.savemat('Raster_COBA.mat', {'Raster_COBA': Raster[:300]})
"""