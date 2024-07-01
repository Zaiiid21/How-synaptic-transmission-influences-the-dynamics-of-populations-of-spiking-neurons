# -*- coding: utf-8 -*-
"""
Created on Mon May 20 17:41:45 2024

@author: zaidh
"""

import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.image as mpimg

# Definir el número de neuronas
num_neurons = 8

# Crear un grafo completo (todos a todos)
G = nx.complete_graph(num_neurons)

# Cargar la imagen de la neurona
neuron_image = mpimg.imread(r'C:\Users\zaidh\Desktop\TFG\Figures\neuron.png')

# Dibujar el grafo
plt.figure(figsize=(8, 8),dpi=300)
pos = nx.spring_layout(G)  # Posiciones de los nodos

# Dibujar las aristas (edges)
nx.draw_networkx_edges(G, pos, alpha=0.5)

# Dibujar los nodos como imágenes
ax = plt.gca()

# Definir el tamaño de la imagen
img_size = 0.23  # Ajusta el tamaño según sea necesario

for n in G.nodes:
    (x, y) = pos[n]
    img = OffsetImage(neuron_image, zoom=img_size)
    ab = AnnotationBbox(img, (x, y), frameon=False)
    ax.add_artist(ab)

# Ajustar los límites del gráfico
plt.xlim(-1.1, 1.1)
plt.ylim(-1.1, 1.1)

# Quitar los ejes para una visualización más limpia
plt.axis('off')

# Título del gráfico
plt.title("Population of All-to-all Coupled Neurons", fontsize=20, fontname='Times New Roman')

# Mostrar el gráfico
plt.show()
