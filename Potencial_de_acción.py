import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import pandas as pd

def action_potential(t):
    if t < 0.33:
        return -70 + 910 * (t ** 2)
    elif t < 0.67:
        return 6800 * np.exp(-12.7 * t) - 80
    else:
        return 10 / (1.15 + np.exp(-50 * (t - 0.743))) - 78.7

# Configuración de fuente global
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman']
rcParams['font.size'] = 16  # Tamaño de fuente más grande para mejor visibilidad

# Vector de tiempo de 0 a 1 ms, con 300 puntos
t = np.linspace(0, 0.9, 3000)

# Aplicar la función a cada punto de tiempo
voltage = np.vectorize(action_potential)(t)
voltage = pd.Series(voltage)  
voltage = voltage.rolling(window=200).mean()

# Graficar
plt.figure(figsize=(8, 6))
plt.plot(t, voltage, label='Membrane Potential', color='blue', linewidth=2.5)  # Línea más gruesa y azul
plt.title('Action Potential Simulation', fontsize=26)
plt.ylabel('Membrane Potential (mV)', fontsize=18)
plt.xlabel('Time (ms)', fontsize=18)
plt.axhline(y=-70, color='gray', linestyle='--', label='Resting Potential (-70 mV)')
plt.legend()
plt.savefig('action_potential', dpi=300)  # Ajusta dpi para mayor resolución
plt.show()
