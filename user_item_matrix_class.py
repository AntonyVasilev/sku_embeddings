from typing import Dict

import pandas as pd
from scipy.sparse import csr_matrix


class UserItemMatrix:
    def __init__(self, sales_data: pd.DataFrame):
        """Class initialization. You can make necessary
        calculations here.

        Args:
            sales_data (pd.DataFrame): Sales dataset.

        Example:
            sales_data (pd.DataFrame):

                user_id  item_id  qty   price
            0        1      118    1   626.66
            1        1      285    1  1016.57
            2        2     1229    3   518.99
            3        4     1688    2   940.84
            4        5     2068    1   571.36
            ...

        """
        self._sales_data = sales_data.copy()

        self._user_count = self._sales_data.user_id.nunique(dropna=True)
        self._item_count = self._sales_data.item_id.nunique(dropna=True)

        self._user_map = {user_: i for i, user_ in
                          enumerate(sorted(self._sales_data.user_id.unique()))}
        self._item_map = {item_: i for i, item_ in
                          enumerate(sorted(self._sales_data.item_id.unique()))}

        _rows = self._sales_data.user_id.map(self._user_map).to_numpy()
        _cols = self._sales_data.item_id.map(self._item_map).to_numpy()
        _data = self._sales_data.qty.to_numpy()

        self._matrix = csr_matrix((_data, (_rows, _cols)), shape=(self._user_count, self._item_count))

    @property
    def user_count(self) -> int:
        """
        Returns:
            int: the number of users in sales_data.
        """
        return self._user_count

    @property
    def item_count(self) -> int:
        """
        Returns:
            int: the number of items in sales_data.
        """
        return self._item_count

    @property
    def user_map(self) -> Dict[int, int]:
        """Creates a mapping from user_id to matrix rows indexes.

        Example:
            sales_data (pd.DataFrame):

                user_id  item_id  qty   price
            0        1      118    1   626.66
            1        1      285    1  1016.57
            2        2     1229    3   518.99
            3        4     1688    2   940.84
            4        5     2068    1   571.36

            user_map (Dict[int, int]):
                {1: 0, 2: 1, 4: 2, 5: 3}

        Returns:
            Dict[int, int]: User map
        """
        return self._user_map

    @property
    def item_map(self) -> Dict[int, int]:
        """Creates a mapping from item_id to matrix rows indexes.

        Example:
            sales_data (pd.DataFrame):

                user_id  item_id  qty   price
            0        1      118    1   626.66
            1        1      285    1  1016.57
            2        2     1229    3   518.99
            3        4     1688    2   940.84
            4        5     2068    1   571.36

            item_map (Dict[int, int]):
                {118: 0, 285: 1, 1229: 2, 1688: 3, 2068: 4}

        Returns:
            Dict[int, int]: Item map
        """
        return self._item_map

    @property
    def csr_matrix(self) -> csr_matrix:
        """User items matrix in form of CSR matrix.

        User row_ind, col_ind as
        rows and cols indecies(mapped from user/item map).

        Returns:
            csr_matrix: CSR matrix
        """
        return self._matrix


if __name__ == '__main__':
    data = pd.read_csv('sales_data.csv')
    matrix = UserItemMatrix(data)

    n_users = matrix.user_count
    n_items = matrix.item_count
    csr_matrix_ = matrix.csr_matrix

    print()
