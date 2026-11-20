import numpy as np
import matplotlib.pyplot as plt
import math

def requested_expression(m_1,m_2):
    if m_1 == 0 or m_2 == 0:
        return math.log2(2)
    return math.log2(2)+m_1/(m_1 + m_2) * math.log2(m_1/(m_1 + m_2)) +m_2/(m_1 + m_2) * math.log2(m_2/(m_1 + m_2))

def P(m_1, m_2):
    return 1-(m_1+m_2)+0.5*((m_1)**(1/2)+(m_2)**(1/2))**2

def H_2(x):
    if x == 0 or x == 1:
        return 0
    return -x * math.log2(x) - (1-x) * math.log2(1-x)

def von_neumann_entropy_upper_bound(m_1, m_2):
    probability = P(m_1, m_2)
    if 1 < probability <= 1 + 1e-11:
        probability = 1.0
    if -1e-11 <= probability < 0:
            probability = 0.0
    if m_1 + m_2 == 0 or m_1 + m_2 == 2:
        return 0.0
    return  1/2*((m_1+m_2)*H_2((1-probability)/(m_1+m_2)) + (2-(m_1+m_2))*H_2((1-probability)/(2-(m_1+m_2))))

# Definizione dei range per m_1 ed m_2
m1_vals = np.linspace(0, 1.0, 100)
m2_vals = np.linspace(0, 1.0, 100)
M1, M2 = np.meshgrid(m1_vals, m2_vals)

# Calcolo delle due superfici 3D
Z_bound = np.vectorize(von_neumann_entropy_upper_bound)(M1, M2)

# Creazione della figura e degli assi 3D
fig = plt.figure(figsize=(10, 7), dpi=100)
ax = fig.add_subplot(111, projection='3d')

# Tracciamento delle due superfici
surf2 = ax.plot_surface(M1, M2, Z_bound, color='red', alpha=0.6, label='Von Neumann Upper Bound')

# Etichette e titolo
ax.set_xlabel(r'$m_1$')
ax.set_ylabel(r'$m_2$')
ax.set_zlabel('Entropies')
ax.set_title('Two outcome quantum measurement (3D)')

# Limiti e griglia
ax.set_xlim(0, 1.0)
ax.set_ylim(0, 1.0)
ax.set_zlim(0, 1.0)

# Per visualizzare correttamente le etichette della legenda con plot_surface
surf2._edgecolors2d = surf2._edgecolor3d
surf2._facecolors2d = surf2._facecolor3d
ax.legend(fontsize=10, loc='upper right')

plt.show()