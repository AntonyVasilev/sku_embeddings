import numpy as np
from scipy.sparse import csr_matrix


def items_embeddings(ui_matrix: csr_matrix, dim: int) -> np.ndarray:
    """Build items embedding using factorization model.
    The order of items should be the same in the output matrix.

    Args:
        ui_matrix (pd.DataFrame): User-Item matrix of size (N, M)
        dim (int): Dimention of embedding vectors

    Returns:
        np.ndarray: Items embeddings matrix of size (M, dim)
    """
    return items_vec
