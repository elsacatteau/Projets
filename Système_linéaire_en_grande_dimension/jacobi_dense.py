import numpy as np
import matplotlib.pyplot as plt
import time

def generate_linear_system(n):
  """
  Generates a linear system with a diagonally dominant matrix A and vector b.

  Args:
    n: Dimension of the system (size of the matrix A).

  Returns:
    A: Coefficient matrix (numpy array).
    b: Right-hand side vector (numpy array).
  """
  ### TODO: Fill in the necessary code
  
  A = np.zeros((n, n))
  for i in range(n):
    for j in range(n):
      if i==j:
        A[i,i]  = 5*(i+1)
      else:
        A[i,j] = -1
  b = np.random.rand(n)

  return A, b

# Example usage:
n = 100  # Dimension of the system
A, b = generate_linear_system(n)


def jacobi_method(A, b, x0, tol=1e-5, max_iter=1000):
  """
  Implements the Jacobi method for solving the linear system Ax = b.

  Args:
    A: Coefficient matrix (numpy array).
    b: Right-hand side vector (numpy array).
    x0: Initial guess for the solution vector (numpy array).
    tol: Tolerance for convergence.
    max_iter: Maximum number of iterations.

  Returns:
    x: Approximate solution vector.
    iterations: Number of iterations performed.
    errors: List of errors between exact and approximate solution at each iteration.
  """
  ### TODO: Review code here
  start_time=time.time()
  n = A.shape[0]
  x = x0.copy()
  errors = []
  for k in range(max_iter):
    x_new = np.zeros_like(x)
    for i in range(n):
        x_new[i]=(1/A[i,i])*(b[i]-np.dot(A[i,:],x)+A[i,i]*x[i])
    error = np.linalg.norm(x_new-x)
    errors.append(error)
    x = x_new
    if error < tol:
        break
  end_time=time.time()
  time_taken=end_time-start_time
  #calcul du rayon spectral
  D=np.diag(np.diag(A))
  T=np.matmul(np.linalg.inv(D),D-A)
  r=np.max(np.absolute(np.linalg.eigvals(T)))
  #matrice dominante
  sum=np.sum(np.absolute(A-D))
  if np.all(((np.absolute(np.diag(A))))>np.sum(np.absolute(A-D),axis=1)): #axis=1 colonne 
    print('La matrice est dominante')
  else:
    print("La matrice n'est pas dominante")
  return x, k + 1, errors, time_taken, r


def plot_error(errors, iterations):
    plt.figure(figsize=(8, 6))
    plt.plot(range(iterations), errors, marker='o', linestyle='-')
    plt.semilogy(range(iterations), errors, marker='o', linestyle='-')  # Use semilogy for log-scale on y-axis
    plt.xlabel("Iterations")
    plt.ylabel("Error estimate")
    plt.title("Error vs Iterations (Jacobi Method)")
    plt.grid(True)
    plt.show()
    
# Example usage:
n = 100
A, b = generate_linear_system(n)  # Generate a linear system
x0 = np.zeros(np.size(b))

# Solve using Jacobi method
x_jacobi, iterations, errors, time, r = jacobi_method(A, b, x0)

# Calculate exact solution
x_exact = np.linalg.solve(A, b)

# Print results
print(f"Iterations: {iterations}")
print(f"Solution Jacobi: {x_jacobi}")
print(f"Exact solution: {x_exact}")
print(f"Temps: {time}") #temps renvoyé 0.5010738372802734
print(f"Rayon spectral: {r}") #pour 500 on a 1.313437094839583 (diverge) et pour 100 on a 0.979170670447409 (converge)
# Plot the error
plot_error(errors, iterations)