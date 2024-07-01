import numpy as np
import matplotlib.pyplot as plt
import time
import math
from scipy.integrate import odeint

#VARIABLES
###############################################################################
N = 1000      # We define the global parameters
vres = 100
vref = -vres/10
ti = 0
Tmax= 50
dt=0.001
ventana = 0.022
tau = int(ventana/dt)
refractory_period = ((2/vres)/dt) # You can adjust this value as needed.
t0 = 10
t2 = 40
I = 6
sn = 1/dt
coupling = 5
delta = 1
x0 = -4
SD = 0.001 #Where I want the Fr to start
n0 = -2 #Where I want the V to start
###############################################################################

#FUNCTIONS AND INITATION
###############################################################################
def lorentz(j,x,gamma,m=N):
    return x+gamma*math.tan((np.pi/2)*(2*j-m-1)/(m+1))

def s(t,spike):
    return spike

def I0(t,ti=t0,tf=t2,Ia=I):
    if t > ti and t < tf:
        return Ia
    else:
        return 0  #We define the applied current to the network

def I(t,mu,spike,J=coupling):
    return mu + J*s(t,spike) + I0(t) 

def f_v(v0,t,mu,spike):
    return (v0**2) + I(t,mu,spike)

tvec=np.arange(ti,Tmax,dt)
v0=np.zeros((N, len(tvec)))
for inicial in range(1,N+1):
    v0[inicial-1][0] = round(lorentz(inicial,n0,SD),5)
    
# Mix values
np.random.shuffle(v0)
    
spike_times= []
[spike_times.append([]) for x in range(N)]
Raster = np.ones((N,len(tvec)))


start_time = time.time()

mu = []

for x in range(1,N+1):
    mu.append(round(lorentz(x,x0,delta),5))
    
# Mix values
np.random.shuffle(mu)

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
            v[j-1] += vref/N #voltage
            v0[i][j]= v0[i][j-1]
            refractory_state[i] -= 1  # Decrease refractory state.
        else:
            v[j-1] += v0[i][j-1]/N #voltage
            v0[i][j]=v0[i][j-1]+dt*f_v(v0[i][j-1],tvec[j-1],mu[i],S[j-1])    #We use Euler method in order to compute the ODE's
            
        #Reset the phases to 0 if they are larger than the threshold
        if v0[i][j]>vres:
            v0[i][j] = v0[i][j]-2*vres
            refractory_state[i] = refractory_period #We enter refractory time.
            #Save the spiking time when the threshold is passed.
    
            spike_times[i].append(tvec[j])
            Raster[i][j] = 0
            
            spike[j] += sn
    if j<tau: #Distributional Dirac delta integral.
        S[j] = (np.sum(spike)/j)/N
    else:
        S[j] = (np.sum(spike[-tau:])/tau)/N

v[-1] = v[-2]
end_time = time.time()
###############################################################################

#%%
"""
plt.figure(figsize=(21,5))
plt.plot(tvec,v0[0],'r-',label="v")
plt.plot(tvec,[I0(x) for x in tvec],label="I(t)")
plt.plot(spike_times[0],np.ones(len(spike_times[0]))*(vr/1.5),"or")
plt.title('One neuron',size=25)
plt.xlabel('Time',size=15)
plt.ylabel('Spikes over time',size=15)
plt.legend()
plt.show()
"""

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
def f_vr(vr,FR,t,x=x0,J=coupling):
    return (vr**2) + x + J*FR + I0(t) -(np.pi**2)*FR**2

def f_rr(FR,vr,gamma=delta):
    return delta/np.pi + 2*FR*vr

vr = np.zeros((1,len(tvec)))
vr = vr[0]
vr[0] = -2
rr = np.zeros((1,len(tvec)))
rr = rr[0]

for j in range(1,len(tvec)): #j represents time step
        vr[j]=vr[j-1]+dt*f_vr(vr[j-1],rr[j-1],tvec[j-1])
        rr[j]=rr[j-1]+dt*f_rr(rr[j-1],vr[j-1])
"""
#%%
#EDOS SIMPLIFICATION
###############################################################################

# Initial conditions.
dt = 0.001
tvec=np.arange(ti,Tmax,dt)
vri = n0
rri = SD
vrri = [vri, rri]


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

"""
plt.figure(figsize=(400,50))
plt.plot(tvec,rr)
plt.title('Firing Rates',size=25)
plt.xlabel('Time',size=15)
plt.ylabel('$r$',size=15)
plt.legend()
plt.show()  


plt.figure(figsize=(400,50))
plt.plot(tvec,vr)
plt.title('Voltage',size=25)
plt.xlabel('Time',size=15)
plt.ylabel('$v$',size=15)
plt.legend()
plt.show()
"""

#%%
import scipy.io as sio

sio.savemat('V_matlabresin.mat', {'V_matlab': v})
sio.savemat('FRr_matlabresin.mat', {'FRr_matlab': rr})
sio.savemat('Vr_matlabresin.mat', {'Vr_matlab': vr})
sio.savemat('S_matlabresin.mat', {'S_matlab': S})
sio.savemat('Raster_matlabresin.mat', {'Raster_matlab': Raster[:300]})




