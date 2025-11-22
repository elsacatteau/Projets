"""
@author: Catteau Elsa, Costantin Perline, Prouzet Charlotte
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Paramètres
alpha = 0.08
beta = 0.02
gamma = 0.01
delta = 0.005
K = 2000

def S(C_T):
    return alpha * C_T * (1 - C_T / K)

# on définit le système avec les 3 équations
def f(t, y):
    CA, CT, CS = y #y = [CA, CT, CS]
    dCA = -S(CT) + beta*CT + delta*CS
    dCT = S(CT) - (beta + gamma + delta)*CT
    dCS = gamma*CT - delta*CS + delta*CT
    system= np.array([dCA, dCT, dCS])
    return system

# Conditions initiales
y0 = [800, 500, 1500]

#Résolution
solution = solve_ivp(f, [0, 100], y0, t_eval=np.linspace(0, 50, 100))
total = solution.y[0]+solution.y[1]+solution.y[2]

# Tracer les résultats
plt.plot(solution.t, solution.y[0], label="C_A : Qté Carbone atmosphère", color="blue")
plt.plot(solution.t, solution.y[1], label="C_T : Qté Carbone arbres", color="green")
plt.plot(solution.t, solution.y[2], label="C_S : Qté Carbone sols", color="orange")
plt.plot(solution.t, total, label="C_A + C_T + C_S : Qté totale", color="red")
plt.title("Résolution avec la bibliothèque SciPy")
plt.xlabel('Temps')
plt.ylabel('Quantité de carbone')
plt.legend()
plt.ylim(0,3000)
plt.xlim(0,50)
plt.grid()
#plt.savefig("verification.png")
plt.show()