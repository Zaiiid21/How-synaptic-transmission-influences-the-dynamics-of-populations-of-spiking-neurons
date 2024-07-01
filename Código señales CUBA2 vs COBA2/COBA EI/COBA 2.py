# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 01:19:35 2024

@author: zaidh
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import math
from scipy.integrate import odeint

#VARIABLES
###############################################################################
N = 10000   # We define the global parameters
vres = 100
vref = 0
ti = 0
Tmax= 50
dt=0.001
ventana = 0.022  #0.04
tau = int(ventana/dt)
refractory_period = ((2/vres)/dt) #200  # You can adjust this value as needed.
sn = 1/dt

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
n0e = -5 #Where I want the Ve to start
SDi = 0 #Where I want the Fri to start
n0i = -5 #Where I want the Vi to start
#%%
###############################################################################

#FUNCTIONS AND INITATION
###############################################################################
def lorentz(j,x,gamma,m=N):
    return x+gamma*math.tan((np.pi/2)*(2*j-m-1)/(m+1))

def s(t,spike):
    return spike

def I0(t, Iae=Iexc,ti=t0,tf=t2):
    if t > ti and t < tf:
        return Iae #Ia*np.sin((np.pi/20)*t)
    else:
        return 0  #We define the applied current to the network

def f_v(v0,t,mu,g,E,spike,I=Iexc):
    return (v0**2) + mu +s(t,spike)*(g+I0(t))*(E-v0) 

tvec=np.arange(ti,Tmax,dt)
v0e=np.zeros((N, len(tvec)))
v0i=np.zeros((N, len(tvec)))
for inicial in range(1,N+1):
    v0e[inicial-1][0] = round(lorentz(inicial,n0e,SDe),5)
for inicial in range(1,N+1):
    v0i[inicial-1][0] = round(lorentz(inicial,n0i,SDi),5)
    
# Mix values
np.random.shuffle(v0e)
np.random.shuffle(v0i)
    
spike_times_e= []
spike_times_i= []
[spike_times_e.append([]) for x in range(N)]
[spike_times_i.append([]) for x in range(N)]
Raster_e = np.ones((N,len(tvec)))
Raster_i = np.ones((N,len(tvec)))


start_time = time.time()

mu_e = []
mu_i = []

for x in range(1,N+1):
    mu_e.append(round(lorentz(x,x0e,delta),5))
    mu_i.append(round(lorentz(x,x0i,delta),5))
    
# Mix values
np.random.shuffle(mu_e)
np.random.shuffle(mu_i)

spike_e = [0]
spike_i = [0]
S_e = np.zeros((1,len(tvec)))
S_e = S_e[0]
S_i = np.zeros((1,len(tvec)))
S_i = S_i[0]
ve = np.zeros((1,len(tvec)))
ve = ve[0]
vi = np.zeros((1,len(tvec)))
vi = vi[0]

refractory_state_e = np.zeros(N)
refractory_state_i = np.zeros(N)
###############################################################################
#%%

#POPULATION INTEGRATION
###############################################################################

for j in range(1,len(tvec)): #j represents time step.
    spike_e.append(0)
    spike_i.append(0)
    for i in range(N): #i represents neuron.
        if refractory_state_e[i] > 0:
            ve[j-1] += vref/N #voltage
            v0e[i][j]= v0e[i][j-1]
            refractory_state_e[i] -= 1  # Decrease refractory state.
        else:
            ve[j-1] += v0e[i][j-1]/N #voltage
            v0e[i][j]=v0e[i][j-1]+dt*f_v(v0e[i][j-1],tvec[j-1],mu_e[i],g,Ei,S_i[j-1])    #We use Euler method in order to compute the ODE's

        if refractory_state_i[i] > 0:
            vi[j-1] += vref/N #voltage
            v0i[i][j]= v0i[i][j-1]
            refractory_state_i[i] -= 1  # Decrease refractory state.
        else:
            vi[j-1] += v0i[i][j-1]/N #voltage
            v0i[i][j]=v0i[i][j-1]+dt*f_v(v0i[i][j-1],tvec[j-1],mu_i[i],g,Ee,S_e[j-1])    #We use Euler method in order to compute the ODE's       
        
        #Reset the phases to 0 if they are larger than the threshold
        if v0e[i][j]>vres:
            v0e[i][j] = v0e[i][j]-2*vres
            refractory_state_e[i] = refractory_period #We enter refractory time.
            #Save the spiking time when the threshold is passed.
            
            spike_times_e[i].append(tvec[j])
            Raster_e[i][j] = 0
            
            spike_e[j] += sn
        #Reset the phases to 0 if they are larger than the threshold
        if v0i[i][j]>vres:
            v0i[i][j] = v0i[i][j]-2*vres
            refractory_state_i[i] = refractory_period #We enter refractory time.
            #Save the spiking time when the threshold is passed.
            
            spike_times_i[i].append(tvec[j])
            Raster_i[i][j] = -1
            
            spike_i[j] += sn
    if j<tau: #Distributional Dirac delta integral.
        S_e[j] = (np.sum(spike_e)/j)/N
        S_i[j] = (np.sum(spike_i)/j)/N
    else:
        S_e[j] = (np.sum(spike_e[-tau:])/tau)/N 
        S_i[j] = (np.sum(spike_i[-tau:])/tau)/N 

ve[-1] = ve[-2]
vi[-1] = vi[-2]
end_time = time.time()
###############################################################################

#%%
print("Tiempo de ejecución:", end_time - start_time, "segundos")
#%%
#EDOS SIMPLIFICATION
###############################################################################

# Initial conditions.
tvec=np.arange(ti,Tmax,dt)
vrie = n0e
rrie = SDe
vrii = n0i
rrii = SDi
vrri = [vrie, rrie, vrii, rrii]


def f_vrr(vrr, t,xi=x0i,xe=x0e,g=g,delta=delta,Ee=Ee,Ei=Ei):
    vre, rre, vri, rri = vrr #initial conditions
    dvredt = (vre**2) + xe + (g+I0(t))*rri*(Ei-vre) -(np.pi**2)*rre**2 #vr ODE
    drredt = delta/np.pi + 2*rre*vre -(g+I0(t))*rre*rri #rr ODE
    dvridt = (vri**2) + xi + (g+I0(t))*rre*(Ee-vri) -(np.pi**2)*rri**2 
    drridt =delta/np.pi + 2*rri*vri -(g+I0(t))*rri*rre
    return [dvredt, drredt, dvridt, drridt]

vrr = odeint(f_vrr, vrri, tvec)

# We define the X and Y vectors from the result.
vre = vrr[:, 0]
rre = vrr[:, 1]
vri = vrr[:, 2]
rri = vrr[:, 3]
#%%

plt.plot(tvec,rre)
plt.plot(tvec,S_e)
plt.title('Firing Rates e',size=25)
plt.xlabel('Time',size=15)
plt.ylabel('$r_e$',size=15)
plt.legend()
plt.show()  

plt.plot(tvec,rri)
plt.plot(tvec,S_i)
plt.title('Firing Rates i',size=25)
plt.xlabel('Time',size=15)
plt.ylabel('$r_i$',size=15)
plt.legend()
plt.show()  

plt.plot(tvec,vre)
plt.plot(tvec,ve)
plt.title('Voltage',size=25)
plt.xlabel('Time',size=15)
plt.ylabel('$v_e$',size=15)
plt.legend()
plt.show()

plt.plot(tvec,vri)
plt.plot(tvec,vi)
plt.title('Voltage',size=25)
plt.xlabel('Time',size=15)
plt.ylabel('$v_i$',size=15)
plt.legend()
plt.show()

#%%

import scipy.io as sio

sio.savemat('FRe_COBA2.mat', {'FRe_matlab': S_e})
sio.savemat('Ve_COBA2.mat', {'Ve_matlab': ve})
sio.savemat('FRi_COBA2.mat', {'FRi_matlab': S_i})
sio.savemat('Vi_COBA2.mat', {'Vi_matlab': vi})
sio.savemat('FRre_COBA2.mat', {'FRre_matlab': rre})
sio.savemat('Vre_COBA2.mat', {'Vre_matlab': vre})
sio.savemat('FRri_COBA2.mat', {'FRri_matlab': rri})
sio.savemat('Vri_COBA2.mat', {'Vri_matlab': vri})
sio.savemat('Rastere_COBA2.mat', {'Rastere_matlab': Raster_e[:300]})
sio.savemat('Rasteri_COBA2.mat', {'Rasteri_matlab': Raster_i[:300]})
