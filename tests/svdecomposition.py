import numpy as np

def reduce_dimensionality_svd(matrix, q):
    """
    Reduces the number of column features of a NumPy array by a factor of q.

    Args:
        matrix (np.ndarray): The input matrix with shape (rows, original_cols).
        q (int): The factor by which to reduce the number of columns.
                 Must be a factor of the original number of columns.

    Returns:
        np.ndarray: The transformed matrix with reduced columns.
    """
    rows, original_cols = matrix.shape
    if original_cols % q != 0:
        raise ValueError("Original columns must be divisible by q.")

    # Number of components to keep
    k = original_cols // q

    # Perform Singular Value Decomposition
    # U, S, and Vt are automatically sorted by singular value magnitude
    U, S, Vt = np.linalg.svd(matrix, full_matrices=False)

    # Reconstruct the matrix using the top k components
    S_reduced = np.diag(S[:k])
    Vt_reduced = Vt[:k, :]

    transformed_matrix = U[:, :k] @ S_reduced @ Vt_reduced

    return transformed_matrix

# Example usage:
# A matrix with 10 rows and 6 features (columns)
original_matrix = np.random.rand(10, 6)

# Reduce the columns by a factor of 2 (6 -> 3 columns)
q = 2
reduced_matrix = reduce_dimensionality_svd(original_matrix, q)

print("Original matrix shape:", original_matrix.shape)
print("Reduced matrix shape:", reduced_matrix.shape)
print("\nOriginal matrix:")
print(original_matrix)
print("\nReduced matrix (approximated):")
print(reduced_matrix)

