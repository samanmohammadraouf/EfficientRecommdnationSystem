import numpy as np
import time

def generate_diagonal_block_matrix(size):
    matrix = np.zeros((size, size))
    
    if size % 2 == 0: 
        for i in range(0, size, 2):
            x1, x2 = np.random.rand(2) * (1 - 1e-10) + 1e-10
            y = np.random.rand() * (1 - 1e-10) + 1e-10
            matrix[i:i+2, i:i+2] = [[x1, y], [y, x2]]
    else:
        for i in range(0, size - 1, 2):
            x1, x2 = np.random.rand(2) * (1 - 1e-10) + 1e-10
            y = np.random.rand() * (1 - 1e-10) + 1e-10
            matrix[i:i+2, i:i+2] = [[x1, y], [y, x2]]
        matrix[-1, -1] = np.random.rand() * (1 - 1e-10) + 1e-10

    return matrix


def generate_eigenvectors(matrix):
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    return eigenvectors

def generate_permutation_matrix(size):
    permutation = np.random.permutation(size)
    P = np.zeros((size, size), dtype=int)
    for i in range(size):
        P[i, permutation[i]] = 1
    return P

def generate_keys(n_users, n_items):
    start_time = time.time()

    S1 = generate_diagonal_block_matrix(n_users)
    S2 = generate_diagonal_block_matrix(n_items)

    E1 = generate_eigenvectors(S1)
    E2 = generate_eigenvectors(S2)

    P1 = generate_permutation_matrix(n_users)
    P2 = generate_permutation_matrix(n_items)

    K1 = np.dot(E1, P1)
    K2 = np.dot(E2, P2)
    
    epsilon = np.random.rand()

    execution_time = time.time() - start_time
    print(f"Key generation completed in {execution_time} seconds.")

    return K1, K2, epsilon

