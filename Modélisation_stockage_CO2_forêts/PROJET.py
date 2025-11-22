import numpy as np
import matplotlib.pyplot as plt
import time #pour comparer les temps des méthodes

#Paramètres
alpha = 0.08
beta = 0.02
gamma = 0.01
delta = 0.005
K = 2000

T = 50 # temps final (en années)
N = 100 # nombre de points
h = T/N #pas

t=np.linspace(0, T, N)

#Initialisation
CA = np.zeros(N)
CT = np.zeros(N)
CS = np.zeros(N)

#Conditions initiales
CA[0]=800 #Qté actuelle de carbone stockée dans l'atmosphère
CT[0]=500 #Qté actuelle de carbone stockée dans les arbres 
CS[0]=1500 #Qté actuelle de carbone stockée dans les sols

#Définition du taux de séquestration du carbone dans les arbres
def S(CT):
    return alpha * CT * (1 - CT / K)

#Dérivée de S(CT)
def dS(CT):
    return alpha * (1 - (2 / K) * CT)

#Méthode Euler implicite
def G(CT_next, CT):
    return CT_next - CT - h * (S(CT_next) - (beta+delta+gamma)*CT_next)

def dG(CT_next):
    return 1 - h * (dS(CT_next) - (beta + delta + gamma))

def newton(CT, eps=1e-6, Nmax=50):
    x=CT
    k=0 #nombre d'itérations
    erreur=abs(G(x, CT))
    while erreur > eps and k < Nmax: #jusqu'aux conditions d'arrêt
        x=x-G(x,CT)/dG(x)
        k=k+1
        erreur=abs(G(x, CT))
    return x

#Résolution en utilisant Newton dans Euler Implicite
def resolution():
    start_time_EI = time.time() 
    for i in range(1, N):
        CT_next=newton(CT[i-1])
        CS_next=(CS[i-1] + h*(gamma + delta)*CT_next) / (1 + h*delta) 
        CA_next=CA[i-1] + h*(-S(CT_next) + beta*CT_next + delta*CS_next)
        CT[i]=CT_next
        CS[i]=CS_next
        CA[i]=CA_next
        
    end_time_EI = time.time()
    time_taken_EI = end_time_EI - start_time_EI #temps d'exécution de la méthode 
    print(f"Temps écoulé avec la méthode euler implicite: {time_taken_EI}")
    
    return (CT, CS, CA)

(CT, CS, CA)=resolution()

#calcul de la somme pour la vérification
total=CT+CA+CS

#afficher le graphe de l'évolution des trois quantités avec EI
plt.plot(t, CA, label="C_A : Qté Carbone atmosphère", color="blue")
plt.plot(t, CT, label="C_T : Qté Carbone arbres", color="green")
plt.plot(t, CS, label="C_S : Qté Carbone sols", color="orange")
plt.plot(t, total, label="C_A + C_T + C_S : Qté totale", color="red")
plt.xlabel("Temps")
plt.ylabel("Quantité de Carbone")
plt.title("Résolution avec euler implicite (EI)")
plt.legend()
plt.ylim(0,3000)
plt.xlim(0,T)
plt.grid()
plt.show()
#plt.savefig("euler_implicite.png") pour sauvegarder le graphe

# on définit le système avec les 3 équations
def f(t, y):
    CA, CT, CS = y #y = [CA, CT, CS]
    dCA = -S(CT) + beta*CT + delta*CS
    dCT = S(CT) - (beta + gamma + delta)*CT
    dCS = gamma * CT - delta * CS + delta * CT
    system= np.array([dCA, dCT, dCS])
    return system

# Méthode d'Euler explicite
def euler_explicite():
    start_time_EE = time.time()
    n=len(y0)
    yEE = np.zeros((N, n))
    yEE[0] = y0
    for i in range(1,N):
        yEE[i] = yEE[i-1] + h * f(t[i-1], yEE[i-1])
    end_time_EE = time.time()
    time_taken_EE = end_time_EE - start_time_EE
    print(f"Temps écoulé avec la méthode euler explicite: {time_taken_EE}")
    return yEE

# Runge-Kutta d'ordre 2
def runge_kutta2():
    start_time_RK = time.time()
    n=len(y0)
    yRK = np.zeros((N, n))
    yRK[0] = y0
    for i in range(1,N):
        k1 = h*f(t[i-1], yRK[i-1])
        k2 = h*f(t[i-1] + h, yRK[i-1] + h * k1)
        yRK[i] = yRK[i-1] + (1/2) * (k1 + k2)
    end_time_RK = time.time()
    time_taken_RK = end_time_RK - start_time_RK
    print(f"Temps écoulé avec la méthode Runge-Kutta: {time_taken_RK}")
    return yRK

y0 = np.array([CA[0], CT[0], CS[0]])
# Euler explicite
y_EE = euler_explicite()
#Extraction de chaque quantité
CA_EE= y_EE[:, 0]
CT_EE = y_EE[:, 1]
CS_EE = y_EE[:, 2]

#somme pour vérification
total_EE= CA_EE+ CT_EE +CS_EE

# Runge-Kutta 2 (même procédé que pour EE)
y_RK2 = runge_kutta2()
CA_RK2 = y_RK2[:, 0]
CT_RK2 = y_RK2[:, 1]
CS_RK2 = y_RK2[:, 2]

#somme pour vérification
total_RK2 = CA_RK2 + CT_RK2 + CS_RK2

# Pour la résolution avec Euler Explicite
plt.plot(t, CA_EE, label="C_A : Qté Carbone atmosphère", color="blue")
plt.plot(t, CT_EE, label="C_T : Qté Carbone arbres", color="green")
plt.plot(t, CS_EE, label="C_S : Qté Carbone sols", color="orange")
plt.plot(t, total_EE, label="C_A + C_T + C_S : Qté totale", color="red")
plt.title("Résolution avec euler explicite (EE)")
plt.xlabel("Temps")
plt.ylabel("Quantité de carbone")
plt.legend()
plt.ylim(0,3000)
plt.xlim(0,T)
plt.grid()
plt.show()
#plt.savefig("euler_explicite.png")

# Pour la résolution avec Runge-Kutta 2
plt.plot(t, CA_RK2, label="C_A : Qté Carbone atmosphère", color="blue")
plt.plot(t, CT_RK2, label="C_T : Qté Carbone arbres", color="green")
plt.plot(t, CS_RK2, label="C_S : Qté Carbone sols",color="orange")
plt.plot(t, total_RK2, label="C_A + C_T + C_S : Qté totale", color="red")
plt.title("Résolution avec Runge-Kutta ordre 2 (RK2)")
plt.xlabel("Temps")
plt.ylabel("Quantité de carbone")
plt.legend()
plt.ylim(0,3000)
plt.xlim(0,T)
plt.grid()
plt.show()
#plt.savefig("runge_kutta2.png")



