import numpy as np
import scipy.sparse as sparse
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt

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

def symmetric_successive_over_relaxation(A,b,x0, x_exact, tol=1e-5, max_iter=1000,w=1):
    """
    Args:
        A: Sparse coefficient matrix (scipy.sparse.csr_matrix).
        b: Right-hand side vector (numpy array).
        x0: Initial guess for the solution vector (numpy array).
        x_exact : The true solution, used to compute ||x^(k)-x_exact||
        tol: Tolerance for convergence.
        max_iter: Maximum number of iterations.
        w: relaxiation parameter (0<w<2)
    Returns:
        x : Approximate solution vector.
        iterations: Number of iterations performed.
        errors: List of errors between exact and approximate solution at each iteration.
    """
    x=x0.copy()
    D=sparse.diags(A.diagonal())
    L=sparse.tril(A)-D
    U=sparse.triu(A)-D
    C=(1/w)*(D+w*L)@sparse.linalg.inv(D)@(D+(w*U))
    E=D+w*U
    E1=sparse.linalg.inv(E)
    errors=[]
    for k in range(max_iter):
        x_new=E1.dot(b)
        error=np.linalg.norm(x_new-x_exact)
        errors.append(error)
        if error < tol:
            break
        x = x_new
    return x, k + 1, errors

def plot_error(errors, iterations):
    plt.figure(figsize=(8, 6))
    plt.plot(range(iterations), errors, marker='o', linestyle='-')
    plt.semilogy(range(iterations), errors, marker='o', linestyle='-')  # Use semilogy for log-scale on y-axis
    plt.xlabel("Iterations")
    plt.ylabel("Error estimate")
    plt.title("Error vs Iterations (SSOR)")
    plt.grid(True)
    plt.show()

n=100
x0=np.zeros(n)
A, A_dense, b=generate_sparse_tridiagonal_matrix(n)
x_exact=sparse.linalg.inv(A)*b
x, iter, errors=symmetric_successive_over_relaxation(A,b,x0, x_exact)

print(f"Avec SSOR on trouve : x={x}, {iter} iterations, erreurs:{errors}")
plot_error(errors, iter)
