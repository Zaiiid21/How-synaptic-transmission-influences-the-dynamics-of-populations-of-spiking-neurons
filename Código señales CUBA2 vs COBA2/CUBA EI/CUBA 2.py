# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 01:57:51 2023

@author: zaidh
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import math
from scipy.integrate import odeint

#VARIABLES
###############################################################################
N = 10000      # We define the global parameters
vres = 100
vref = -vres/10
ti = 0
Tmax= 50
dt=0.001
ventana = 0.015
tau = int(ventana/dt)
refractory_period = ((4/vres)/dt) #200  # You can adjust this value as needed.
t0 = 10
t2 = 30
Iexc = 5
Iinh = 5
sn = 1/dt
couplingee = 0
couplingie = 10
couplingei = 10
couplingii = 0
delta = 1
x0e = 5
x0i = -5
SDe = 0 #Where I want the Fre to start
n0e = -3 #Where I want the Ve to start
SDi = 0 #Where I want the Fri to start
n0i = -3 #Where I want the Vi to start
#%%
###############################################################################

#FUNCTIONS AND INITATION
###############################################################################
def lorentz(j,x,gamma,m=N):
    return x+gamma*math.tan((np.pi/2)*(2*j-m-1)/(m+1))

def s(t,spike):
    return spike

def I0(t, Iae,ti=t0,tf=t2):
    if t > ti and t < tf:
        return Iae #Ia*np.sin((np.pi/20)*t)
    else:
        return 0  #We define the applied current to the network

def I(t,mu,spike,Iae,J=couplingie):
    return mu + (J+ I0(t,Iae))*s(t,spike) 

def f_v(v0,t,mu,spike,Iae):
    return (v0**2) + I(t,mu,spike,Iae)

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
            v0e[i][j]=v0e[i][j-1]+dt*f_v(v0e[i][j-1],tvec[j-1],mu_e[i],S_i[j-1],Iexc)    #We use Euler method in order to compute the ODE's

        if refractory_state_i[i] > 0:
            vi[j-1] += vref/N #voltage
            v0i[i][j]= v0i[i][j-1]
            refractory_state_i[i] -= 1  # Decrease refractory state.
        else:
            vi[j-1] += v0i[i][j-1]/N #voltage
            v0i[i][j]=v0i[i][j-1]+dt*f_v(v0i[i][j-1],tvec[j-1],mu_i[i],S_e[j-1],Iinh)    #We use Euler method in order to compute the ODE's       
        
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
            
            spike_i[j] -= sn
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


def f_vrr(vrr, t,xi=x0i,xe=x0e,Jee=couplingee,Jei=couplingei,Jie=couplingie,Jii=couplingii,gamma=delta):
    vre, rre, vri, rri = vrr #initial conditions
    dvredt = (vre**2) + xe + Jee*rre - (Jie+ I0(t,Iexc))*rri -(np.pi**2)*rre**2 #vr ODE
    drredt = delta/np.pi + 2*rre*vre #rr ODE
    dvridt = (vri**2) + xi + (Jei+ I0(t,Iinh))*rre - Jii*rri -(np.pi**2)*rri**2
    drridt =delta/np.pi + 2*rri*vri
    return [dvredt, drredt, dvridt, drridt]

vrr = odeint(f_vrr, vrri, tvec)

# We define the X and Y vectors from the result.
vre = vrr[:, 0]
rre = vrr[:, 1]
vri = vrr[:, 2]
rri = vrr[:, 3]
#%%
import scipy.io as sio

sio.savemat('FRe_CUBA2.mat', {'FRe_matlab': S_e})
sio.savemat('Ve_CUBA2.mat', {'Ve_matlab': ve})
sio.savemat('FRi_CUBA2.mat', {'FRi_matlab': -S_i})
sio.savemat('Vi_CUBA2.mat', {'Vi_matlab': vi})
sio.savemat('FRre_CUBA2.mat', {'FRre_matlab': rre})
sio.savemat('Vre_CUBA2.mat', {'Vre_matlab': vre})
sio.savemat('FRri_CUBA2.mat', {'FRri_matlab': rri})
sio.savemat('Vri_CUBA2.mat', {'Vri_matlab': vri})
sio.savemat('Rastere_CUBA2.mat', {'Rastere_matlab': Raster_e[:300]})
sio.savemat('Rasteri_CUBA2.mat', {'Rasteri_matlab': Raster_i[:300]})

#%%
"""
# Crear el histograma
plt.hist(mu_e, bins=np.arange(min(mu_e), max(mu_e) + 1, 1), edgecolor='blue')

# Configuraciones adicionales
plt.xlabel('Valor')
plt.ylabel('Frecuencia')
plt.title('Histograma')

# Mostrar el histograma
plt.show()

# Crear el histograma
plt.hist(mu_i, bins=np.arange(min(mu_i), max(mu_i) + 1, 1), edgecolor='blue')

# Configuraciones adicionales
plt.xlabel('Valor')
plt.ylabel('Frecuencia')
plt.title('Histograma')

# Mostrar el histograma
plt.show()
"""