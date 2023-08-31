import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from implicit.cpu.als import AlternatingLeastSquares

from user_item_matrix_class import UserItemMatrix
from matrix_normalization import Normalization


def items_embeddings(ui_matrix: csr_matrix, dim: int) -> np.ndarray:
    """Build items embedding using factorization model.
    The order of items should be the same in the output matrix.

    Args:
        ui_matrix (pd.DataFrame): User-Item matrix of size (N, M)
        dim (int): Dimention of embedding vectors

    Returns:
        np.ndarray: Items embeddings matrix of size (M, dim)
    """
    als_model = AlternatingLeastSquares(factors=dim, iterations=10)
    als_model.fit(ui_matrix)

    items_vec = als_model.item_factors
    return items_vec


if __name__ == '__main__':
    data = pd.read_csv('data/sales_data.csv')
    matrix_ = UserItemMatrix(data)
    csr_matrix_ = matrix_.csr_matrix

    matrix_norm = Normalization.bm_25(csr_matrix_)

    items_vec_ = items_embeddings(ui_matrix=matrix_norm, dim=64)
