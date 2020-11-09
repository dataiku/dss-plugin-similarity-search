from nns_utils import reshape_values


class DataLoader(object):
    def __init__(self, dataset_name, unique_column, embedding_column):
        self._dataset_name = dataset_name
        self._unique_column = unique_column
        self._embedding_column = embedding_column

    def _load(self):

        df = self._dataset_name.get_dataframe(columns=[self._unique_column, self._embedding_column])
        if self._check_column_uniqueness(df):
            df = df[df[self._embedding_column].notnull()]
            vectors = reshape_values(df[self._embedding_column])
            names = df[self._unique_column].tolist()
            return names, vectors
        else:
            raise ValueError("The values in column {} are not unique".format(self._unique_column))

    def _check_column_uniqueness(self, df):
        """
        Verify that all values in the column are unique, as they will be used as lookups for the indices.
        """
        if len(df[self._unique_column].unique()) == len(df[self._unique_column].tolist()):
            return True
        else:
            return False
