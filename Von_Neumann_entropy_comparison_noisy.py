import numpy as np
import matplotlib.pyplot as plt
import math

def requested_expression(epsilon, d):
    lambda_0 = 1 - epsilon + epsilon / d
    epsilon_log_term = 0 if epsilon == 0 else epsilon * math.log2(epsilon)
    return (1 - epsilon) * math.log2(d) + epsilon_log_term + lambda_0 * math.log2(lambda_0)

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

d = 2
epsilon_vals = np.linspace(0, 1, 100)

plt.figure(figsize=(8.5, 5.5), dpi=100)
plt.plot(epsilon_vals, [requested_expression(eps, d) for eps in epsilon_vals], color='blue', lw=1.5, label=r'State Von Neumann Entropy')
plt.plot(epsilon_vals, [von_neumann_entropy_upper_bound(eps, d) for eps in epsilon_vals], color='red', lw=1.5, label=r'Von Neumann Entropy upper bound')
plt.xlabel(r'$\epsilon$')
plt.ylabel(r'Entropies')
plt.title(f'Noisy quantum measurement (d={d})')
plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.5)
plt.legend(fontsize=10, loc='upper right')
plt.xlim(0, 1)
plt.show()