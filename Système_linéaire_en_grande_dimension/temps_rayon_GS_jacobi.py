import numpy as np
import scipy.sparse as sparse
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
import time

def generate_simple_sparse_tridiagonal_matrix(n, diagonal_value=10, off_diagonal_value=4):
    """
    Generates a sparse tridiagonal matrix, ensuring no overlaps.

    Args:
        n: Dimension of the system (size of the matrix A).
        diagonal_value: Value for the diagonal elements.
        off_diagonal_value: Value for the off-diagonal elements.

    Returns:
        A: Sparse coefficient matrix (scipy.sparse.csr_matrix).
        A_dense: equivalent Dense matrix (numpy array)
        b: Right-hand side vector (numpy array).
    """
    ### TODO: Fill your code here
    main_diag = np.full(n, diagonal_value)
    off_diag = np.full(n-1, off_diagonal_value)

    # Construct sparse matrix
    data = np.concatenate([main_diag,off_diag,off_diag])
    rows = np.concatenate([np.arange(n), np.arange(n-1), np.arange(1,n)]) # might need to use np.concatenate
    cols = np.concatenate([np.arange(n), np.arange(1,n), np.arange(n-1)])
    As = csr_matrix((data, (rows, cols)), shape=(n, n)) # Generates CSR Matrix from COO representation

    # Construct dense matrix for reference
    A_dense = np.zeros((n, n))
    A_dense[n-1,n-1]=diagonal_value
    for i in range(n-1):
        A_dense[i, i] = diagonal_value
        A_dense[i,i+1]=off_diagonal_value
        A_dense[i+1,i]=off_diagonal_value
    b = np.random.rand(n)
    return As, A_dense, b

def generate_sparse_tridiagonal_matrix(n):
    """
    Generates a sparse tridiagonal matrix with the specific values.

    Args:
        n: Dimension of the system (size of the matrix A).

    Returns:
        A: Sparse coefficient matrix (scipy.sparse.csr_matrix).
        b: Right-hand side vector (numpy array).
    """
    ### TODO: Fill your code here.
    h=1/(n+1)
    diagonal_value=2
    off_diagonal_value=-1
    main_diag = np.full(n, diagonal_value)
    off_diag = np.full(n-1, off_diagonal_value)
    data = np.concatenate([main_diag,off_diag,off_diag])
    rows = np.concatenate([np.arange(n), np.arange(n-1), np.arange(1,n)]) # might need to use np.concatenate
    cols = np.concatenate([np.arange(n), np.arange(1,n), np.arange(n-1)])
    A = (1/(h**2))*csr_matrix((data, (rows, cols)), shape=(n, n))

    # Construct dense matrix for reference
    A_dense = np.zeros((n, n))
    A_dense[n-1,n-1]=diagonal_value
    for i in range(n-1):
        A_dense[i, i] = diagonal_value
        A_dense[i,i+1]=off_diagonal_value
        A_dense[i+1,i]=off_diagonal_value
    A_dense=(1/(h**2))*A_dense
    b = np.random.rand(n)
    return A,  A_dense, b



def jacobi_sparse_with_error(A, b, x0, x_exact, tol=1e-5, max_iter=1000):
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
    D1=1/A.diagonal()
    LU=A-sparse.diags(A.diagonal())
    errors = []
    for k in range(max_iter):
        x_new = D1*(b-LU.dot(x))
        error = np.linalg.norm(x_new-x_exact)
        errors.append(error)
        if error < tol:
            break
        x = x_new
    end_time=time.time()
    time_taken=end_time-start_time
    #calcul du rayon spectral
    D=sparse.diags(A.diagonal())
    T=(sparse.linalg.inv(D)).dot(D-A)
    rayon=np.max(np.absolute(np.linalg.eigvals(T.todense())))
    return x, k + 1, errors, time_taken, rayon


def gauss_seidel_sparse_with_error(A, b, x0, x_exact, tol=1e-5, max_iter=1000):
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
    C=sparse.tril(A)  #correspond à D-L
    U=sparse.triu(A)-sparse.diags(A.diagonal())
    errors = []
    C1=sparse.linalg.inv(C)
    for k in range(max_iter):
        x_new = C1*(b-U.dot(x))
        error = np.linalg.norm(x_new-x_exact)
        errors.append(error)
        if error < tol:
            break
        x = x_new
    end_time=time.time()
    time_taken=end_time-start_time
    #calcul du rayon spectral
    D=sparse.diags(A.diagonal())
    T=(sparse.linalg.inv(D)).dot(D-A)
    rayon=np.max(np.absolute(np.linalg.eigvals(T.todense())))
    return x, k + 1, errors, time_taken, rayon

def plot_error(errors, iterations, name):
    plt.figure(figsize=(8, 6))
    plt.plot(range(iterations), errors, marker='o', linestyle='-')
    plt.semilogy(range(iterations), errors, marker='o', linestyle='-')  # Use semilogy for log-scale on y-axis
    plt.xlabel("Iterations")
    plt.ylabel("Error estimate")
    plt.title("Error vs Iterations", name)
    plt.grid(True)
    plt.show()

### TODO: 
# Set up all the important parameters
# Set up all useful plotting tools

n=100
x0 = np.zeros(n)
As1, A_dense1, b1 = generate_simple_sparse_tridiagonal_matrix(n)
As2, A_dense2, b2 =generate_sparse_tridiagonal_matrix(n)

x_exact1 = sparse.linalg.inv(As1) * b1
x_exact2 = sparse.linalg.inv(As2) * b2

# Solve using Jacobi sparse method with error
x_jacobi1, iter_jacobi1, errors_jacobi1, time_jacobi1, rayon_jacobi1 = jacobi_sparse_with_error(As1, b1, x0, x_exact1)
x_jacobi2, iter_jacobi2, errors_jacobi2, time_jacobi2, rayon_jacobi2 = jacobi_sparse_with_error(As2, b2, x0, x_exact2)

# Solve using Gauss Seidel sparse with error
x_gs1, iter_gs1, errors_gs1, time_gs1, rayon_gs1 = gauss_seidel_sparse_with_error(As1, b1, x0, x_exact1)
x_gs2, iter_gs2, errors_gs2, time_gs2, rayon_gs2 = gauss_seidel_sparse_with_error(As2, b2, x0, x_exact2)

# Print results
print(f"On génère A avec generate_simple_sparse_tridiagonal_matrix :")
print(f"Avec Jacobi on trouve : x={x_jacobi1}, {iter_jacobi1} iterations, erreurs:{errors_jacobi1}, Temps: {time_jacobi1}, Rayon Spectral: {rayon_jacobi1}") #Temps renvoyé 0.017526865005493164 et rayon 0.7996130258335907
print(f"Avec Gauss Seidel on trouve : x={x_gs1}, {iter_gs1} iterations, erreurs:{errors_gs1}, Temps: {time_gs1}, Rayon Spectral: {rayon_gs1}") #Temps renvoyé: 0.11366629600524902

print(f"On génère A avec generate_sparse_tridiagonal_matrix :")
print(f"Avec Jacobi on trouve : x={x_jacobi2}, {iter_jacobi2} iterations, erreurs:{errors_jacobi2}, Temps: {time_jacobi2}, Rayon Spectral: {rayon_jacobi2}") #Temps renvoyé: 0.0601043701171875 et rayon 0.9995162822919886
print(f"Avec Gauss Seidel on trouve : x={x_gs2}, {iter_gs2} iterations, erreurs:{errors_gs2}, Temps: {time_gs2}, Rayon Spectral: {rayon_gs2}") #Temps renvoyé: 0.23833703994750977

#plot_error(errors_jacobi1, iter_jacobi1, 'jacobi_sparse_1')
#plot_error(errors_jacobi2, iter_jacobi2, 'jacobi_sparse_2')
#plot_error(errors_gs1, iter_gs1, 'Gauss_Seidel_1')
#plot_error(errors_gs2, iter_gs2, 'Gauss_Seidel_2')