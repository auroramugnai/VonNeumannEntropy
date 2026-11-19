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

von_neumann_entropy_lower_bound = np.array([0.9887290010809897, 0.8367529151901332, 0.6930627625412251, 0.5582558076171891, 0.4335769485123615, 0.32052968921565217, 0.22056713558325924, 0.13510472830909404, 0.06624451968378331, 0.0549979500792066, 0.044684314656744384, 0.03532162077795942, 0.026937101029298915, 0.019550514891950144, 0.013179169386800638, 0.007839319563883507, 0.003581649978666057, 0.0008900645863816337, 6.074223701801884e-08, 0.0008956926817864641, 0.00358276540737279, 0.007841174213843338, 0.013183244673340866,  0.019552216852861015, 0.02693420218132303, 0.035319310117625236, 0.0446766513984409, 0.05499796474260865, 0.06624297443140836, 0.135104807036301, 0.22056636058619716, 0.32053007949021595, 0.4335760725065869, 0.5582497820088195, 0.6930633516734996, 0.8367516823592247, 0.9887290010867122])
m1 = np.array([0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48, 0.49, 0.5, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59, 0.6, 0.65, 0.70, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0])
m2 = 1-m1
m_1 = np.linspace(0, 1.0, 100)
m_2 = 1.0 - m_1
state_entropy_vals = [
    requested_expression(m_1_val, m_2_val)
    for m_1_val, m_2_val in zip(m_1, m_2)
]
entropy_upper_bound_vals = [
    von_neumann_entropy_upper_bound(m_1_val, m_2_val)
    for m_1_val, m_2_val in zip(m_1, m_2)
]

plt.figure(figsize=(8.5, 5.5), dpi=100)
plt.plot(m_1, state_entropy_vals, color='blue', lw=1.5, label=r'State: $\log_2(d) - S(\rho_\psi)$')
plt.plot(m_1, entropy_upper_bound_vals, color='red', lw=1.5, label=r'Measurement UB: $ H_2\left(1-P_{guess}^*\right)$')
plt.plot(m1, von_neumann_entropy_lower_bound, marker='.', color='green', lw=1.5, label=r'Measurement numerical LB')
plt.xlabel(r'$m_1$')
plt.ylabel(r'Entropies')
plt.title(f'Two outcome quantum measurement ($m_2=1-m_1$)')
plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.5)
plt.legend(fontsize=10, loc='upper right')
plt.xlim(0, 1)
plt.show()