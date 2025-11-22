import numpy as np
import matplotlib.pyplot as plt
import time

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
  if np.all(((np.absolute(np.diag(A))))>np.sum(np.absolute(A-D),axis=1)): #axis=1 colonne 
    print('La matrice est dominante')
  else:
    print("La matrice n'est pas dominante")
  return x, k + 1, errors, time_taken, r

def gauss_seidel_dense_with_error(A, b, x0, x_exact, tol=1e-5, max_iter=1000):
    ### TODO: 
    """
    Args:
        A: Sparse coefficient matrix (scipy.sparse.csr_matrix).
        b: Right-hand side vector (numpy array).
        x0: Initial guess for the solution vector (numpy array).
        x_exact : The true solution, used to compute ||x^(k)-x_exact||
        tol: Tolerance for convergence.
        max_iter: Maximum number of iterations.
    Returns:
        x : Approximate solution vector.
        iterations: Number of iterations performed.
        errors: List of errors between exact and approximate solution at each iteration.
    """
    start_time=time.time()
    x=x0.copy()
    C=np.tril(A)  #correspond à D-L
    U=np.triu(A)-np.diag(np.diag(A))
    errors = []
    C1=np.linalg.inv(C)
    for k in range(max_iter):
        x_new = C1.dot(b-np.dot(U,x))
        error = np.linalg.norm(x_new-x_exact)
        errors.append(error)
        if error < tol:
            break
        x = x_new
    end_time=time.time()
    time_taken=end_time-start_time
    return x, k + 1, errors, time_taken



def plot_error(errors, iterations, name):
    plt.figure(figsize=(8, 6))
    plt.plot(range(iterations), errors, marker='o', linestyle='-')
    plt.semilogy(range(iterations), errors, marker='o', linestyle='-')  # Use semilogy for log-scale on y-axis
    plt.xlabel("Iterations")
    plt.ylabel("Error estimate")
    plt.title("Error vs Iterations"+ name)
    plt.grid(True)
    plt.show()
    
# Example usage:
n = 100
A3=np.array([[4,1,1],[2,-9,0],[0,-8,-6]])
A4=np.array([[7,6,9],[4,5,-4],[-7,-3,8]])
b3=np.array([6,-7,-14])
b4=np.array([22,5,-2])
x0 = np.zeros(np.size(b3))
# Calculate exact solution
x_exact1 = np.linalg.solve(A3, b3)
x_exact2 = np.linalg.solve(A4, b4)

# Solve using Jacobi method
x_jacobi1, iterations_jacobi1, errors_jacobi1, time_jacobi1, r_1 = jacobi_method(A3, b3, x0)
x_jacobi2, iterations_jacobi2, errors_jacobi2, time_jacobi2, r_2 = jacobi_method(A4, b4, x0)
x_gs1, iterations_gs1, errors_gs1, time_gs1=gauss_seidel_dense_with_error(A3, b3, x0, x_exact1)
x_gs2, iterations_gs2, errors_gs2, time_gs2=gauss_seidel_dense_with_error(A4, b4, x0, x_exact2)

# Print results
print(f"Pour la matrice A3 (Jacobi), Solution Jacobi: {x_jacobi1}, Iterations: {iterations_jacobi1}, Temps: {time_jacobi1}") #temps renvoyé 0.0003616809844970703
print(f"Pour la matrice A4 (Gauss Seidel), Solution GS: {x_gs1}, Iterations: {iterations_gs1}, Temps: {time_gs1}")
print(f"Pour la matrice A3, Rayon spectral: {r_1}, Exact solution: {x_exact1}") 
print(f"Pour la matrice A4 (Jacobi), Solution Jacobi: {x_jacobi2}, Iterations: {iterations_jacobi2}, Temps: {time_jacobi2}")#temps renvoyé 0.016640663146972656
print(f"Pour la matrice A4 (Gauss Seidel), Solution GS: {x_gs2}, Iterations: {iterations_gs2}, Temps: {time_gs2}")
print(f"Pour la matrice A4, Exact solution: {x_exact2}, Rayon spectral: {r_2}")

# Plot the error
plot_error(errors_jacobi1, iterations_jacobi1, "Matrice A3 (Jacobi)")
plot_error(errors_jacobi2, iterations_jacobi2, "Matrice A4 (Jacobi)")
plot_error(errors_gs1, iterations_gs1, "Matrice A3 (Gauss Seidel)")
plot_error(errors_gs2, iterations_gs2, "Matrice A4 (Gauss Seidel)")