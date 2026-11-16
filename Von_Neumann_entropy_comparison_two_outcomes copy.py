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
    return  1/2*((m_1+m_2)*H_2((1-probability)/(m_1+m_2)) + (2-(m_1+m_2))*H_2((1-probability)/(2-(m_1+m_2))))

m_1 = np.linspace(0, 0.5, 100)
m_2 = 0.5

plt.figure(figsize=(8.5, 5.5), dpi=100)
plt.plot(m_1, [requested_expression(m_1_val, m_2) for m_1_val in m_1], color='blue', lw=1.5, label=r'State: $\log_2(d) - S(\rho_\psi)$')
plt.plot(m_1, [von_neumann_entropy_upper_bound(m_1_val, m_2) for m_1_val in m_1], color='red', lw=1.5, label=r'Measurement UB: $\frac{1}{2}\left[\text{Tr}[M_1] H_2\left(\frac{1-P_{guess}^*}{\text{Tr}[M_1]}\right)+ (2-\text{Tr}[M_1]) H_2\left(\frac{1-P_{guess}^*}{2-\text{Tr}[M_1]}\right) \right]$')
plt.xlabel(r'$m_1$')
plt.ylabel(r'Entropies')
plt.title(f'Two outcome quantum measurement ($m_2={m_2}$)')
plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.5)
plt.legend(fontsize=10, loc='upper right')
plt.xlim(0, 0.5)
plt.show()