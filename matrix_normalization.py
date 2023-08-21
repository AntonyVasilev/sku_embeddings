from scipy.sparse import csr_matrix
from user_item_matrix_class import UserItemMatrix
import pandas as pd


class Normalization:
    @staticmethod
    def by_column(matrix: csr_matrix) -> csr_matrix:
        """Normalization by column

        Args:
            matrix (csr_matrix): User-Item matrix of size (N, M)

        Returns:
            csr_matrix: Normalized matrix of size (N, M)
        """
        norm_matrix = matrix.multiply(1 / matrix.sum(axis=0)).tocsr()
        return norm_matrix

    @staticmethod
    def by_row(matrix: csr_matrix) -> csr_matrix:
        """Normalization by row

        Args:
            matrix (csr_matrix): User-Item matrix of size (N, M)

        Returns:
            csr_matrix: Normalized matrix of size (N, M)
        """
        norm_matrix = matrix.multiply(1 / matrix.sum(axis=1)).tocsr()
        return norm_matrix

    @staticmethod
    def tf_idf(matrix: csr_matrix) -> csr_matrix:
        """Normalization using tf-idf

        Args:
            matrix (csr_matrix): User-Item matrix of size (N, M)

        Returns:
            csr_matrix: Normalized matrix of size (N, M)
        """
        TF = matrix.multiply(1 / matrix.sum(axis=1)).tocsr()
        IDF = csr_matrix(matrix.shape[0] / (matrix > 0).sum(axis=0)).log1p()
        norm_matrix = TF.multiply(IDF).tocsr()
        return norm_matrix

    @staticmethod
    def bm_25(
        matrix: csr_matrix, k1: float = 2.0, b: float = 0.75
    ) -> csr_matrix:
        """Normalization based on BM-25

        Args:
            matrix (csr_matrix): User-Item matrix of size (N, M)

        Returns:
            csr_matrix: Normalized matrix of size (N, M)
        """
        TF = matrix.multiply(1 / matrix.sum(axis=1)).tocsr()
        IDF = csr_matrix(matrix.shape[0] / (matrix > 0).sum(axis=0)).log1p()
        sub_numerator = csr_matrix(matrix.sum(axis=1) / matrix.sum(axis=1).mean()).multiply(b)
        sub_numerator.data += (1 - b)
        sub_numerator = sub_numerator.multiply(k1)

        denominator = sub_numerator.multiply(TF.power(-1))

        # denominator = (
        #     (csr_matrix(k1 * (1 - b + csr_matrix(matrix.sum(axis=1) /
        #                                          (matrix > 0).sum(axis=1).mean()).multiply(b).data)))
        #     .multiply(TF.power(-1)).tocsr()
        # )
        denominator.data += 1
        fraction = denominator.power(-1).multiply(k1 + 1)
        norm_matrix = fraction.multiply(IDF)
        return norm_matrix


if __name__ == '__main__':
    data = pd.read_csv('data/sales_data.csv')
    matrix_ = UserItemMatrix(data)
    csr_matrix_ = matrix_.csr_matrix

    # norm_cols = Normalization.by_column(csr_matrix_)
    # norm_rows = Normalization.by_row(csr_matrix_)
    # tf_idf_ = Normalization.tf_idf(csr_matrix_)
    bm_25_ = Normalization.bm_25(csr_matrix_)

    print()
