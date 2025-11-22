import numpy as np
import matplotlib.pyplot as plt

#Paramètres avec des valeurs aléatoires pour le moment
alpha =0.08
beta = 0.02
gamma = 0.01
delta = 0.01
K =2000


T=50
N=100
h=(T/N)



t=np.linspace(0, T, N)

#Initialisation
CA = np.zeros(N)
CT = np.zeros(N)
CS = np.zeros(N)

#Conditions initiales
CA[0]=750 #Qté actuelle de carbone stockée dans l'atmosphère
CT[0]=500 #Qté actuelle de carbone stockée dans les arbres
CS[0]=1500 #Qté actuelle de carbone stockée dans les sols

def S(CT, alpha, K):
    return alpha * CT * (1 - CT / K)

def dS(CT, alpha, K):
    return alpha * (1 - (2 / K) * CT)

#Méthode Euler implicite
def G(CT_next, CT, alpha, beta, gamma, delta, K):
    return CT_next - CT - h * (S(CT_next, alpha, K) - (beta+delta+gamma)*CT_next)

def dG(CT_next, alpha, beta, gamma, delta, K):
    return 1 - h * (dS(CT_next, alpha, K) - (beta + delta + gamma))

def newton(CT, alpha, beta, gamma, delta, K, eps=1e-6, Nmax=50):
    x=CT
    k=0
    erreur=abs(G(x, CT, alpha, beta, gamma, delta, K))
    while erreur > eps and k < Nmax:
        x=x-G(x,CT, alpha, beta, gamma, delta, K)/dG(x, alpha, beta, gamma, delta, K)
        k=k+1
        erreur=abs(G(x, CT, alpha, beta, gamma, delta, K))
    return x

#On fait varier alpha et on regarde l'évolution de la Qté de carbone dans l'atmosphère


def resolution(alpha, beta, gamma, delta, K):
    for i in range(1, N):
        CT_next=newton(CT[i-1], alpha, beta, gamma, delta, K)
        CS_next=(CS[i-1] + h*(gamma + delta)*CT_next) / (1 + h*delta)
        CA_next=CA[i-1] + h*(-S(CT_next, alpha, K) + beta*CT_next + delta*CS_next)

        CT[i]=CT_next
        CS[i]=CS_next
        CA[i]=CA_next

    return (CA, CT, CS)



def plot_parametre():
    delta_val = np.arange(0, 0.25, 0.01)
    plt.figure(figsize=(12, 8))
    for delta in delta_val:
        (CA, CT, CS)=resolution(alpha, beta, gamma, delta, K)
        plt.semilogy(t, CS, label='delta= ' + str(round(delta, 3)))
    plt.title("Qté de Carbonne dans les sols en fonction de delta")
    plt.xlabel("Temps")
    plt.ylabel("Qté de carbone")
    plt.legend(loc='lower left')
    plt.grid(True)
    plt.show()


plot_parametre()
