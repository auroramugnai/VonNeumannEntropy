import numpy as np
import matplotlib.pyplot as plt
import math

# Define the state von neumann entropy expression for rho = (1-epsilon)|psi><psi| + epsilon/d * I_d
def requested_expression(epsilon, d):
    lambda_0 = 1 - epsilon + epsilon / d
    epsilon_log_term = 0 if epsilon == 0 else epsilon * math.log2(epsilon)
    epsilon_log_term_1 = 0 if epsilon == 0 else (epsilon/d) * math.log2((epsilon/d))
    return (1 - epsilon) * math.log2(d) + epsilon_log_term + lambda_0 * math.log2(lambda_0) - epsilon_log_term_1


def P(x, d):
    return (1/d)*((1-x+((x)/(d)))**(((1)/(2)))+(d-1) * ((((x)/(d)))**(((1)/(2)))))**(2)

def H_2(x):
    if x == 0 or x == 1:
        return 0
    return -x * math.log2(x) - (1-x) * math.log2(1-x)

def von_neumann_entropy_upper_bound(eps,d):
    probability = P(eps, d)
    if 1 < probability <= 1 + 1e-12:
        probability = 1.0
    complement_log_term = 0 if probability == 1 else (1-probability) * math.log2(d-1)
    return H_2(probability) + complement_log_term

von_neumann_entropy_lower_bound = np.array([6.074223701801884e-08, 0.005562894768367244,  0.01955497540393658, 0.03987570960966469, 0.06624450241633717, 0.0982384110746608, 0.13510524093976173, 0.17593706379053575, 0.2205663112553813, 0.2688229884325324, 0.32053007949021595, 0.3755072078853203, 0.43357636725693194, 0.49455363760758075, 0.5582497820088195, 0.6245059694161412, 0.6930633516734996, 0.7638347758216009, 0.8367521549097559, 0.9117368124492957, 0.9887290010867122])
d = 2
epsilon_vals_1 = np.array([1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05, 0.0])
epsilon_vals = np.linspace(0, 1, 100)

plt.figure(figsize=(8.5, 5.5), dpi=100)
plt.plot(epsilon_vals, [requested_expression(eps, d) for eps in epsilon_vals], color='blue', lw=1.5, label=r'State: $\log_2(d) - S(\rho_\psi)$')
plt.plot(epsilon_vals, [von_neumann_entropy_upper_bound(eps, d) for eps in epsilon_vals], color='red', lw=1.5, label=r'Measurement UB: $H_2(P_{guess}^*(\mathcal{M}_d))+(1-P_{guess}^*(\mathcal{M}_d))\log_2(d-1)$')
plt.plot(epsilon_vals_1, von_neumann_entropy_lower_bound, marker='.', color='green', lw=1.5, label=r'Measurement numerical LB')
plt.xlabel(r'$\epsilon$')
plt.ylabel(r'Entropies')
plt.title(f'Noisy quantum measurement (d={d})')
plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.5)
plt.legend(fontsize=10, loc='upper right')
plt.xlim(0, 1)
plt.show()