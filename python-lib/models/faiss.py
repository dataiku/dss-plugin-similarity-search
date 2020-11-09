import faiss
from models.base import BaseNNS
import os
import json


class Faiss(BaseNNS):

    def __init__(self, *args, **kwargs):
        self._index = kwargs.get("faiss_index")
        self._dims = kwargs.get("dims")
        self._lsh_n_bits = int(kwargs.get("faiss_lsh_n_bits", 4))

    def fit_and_save(self, names, vectors, folder):
        """
        Create requested index from params.
        Add vectors and store index.
        Generate configuration file and save alongside the index.
        """
        folder_path = folder.get_path()

        if self._index == "IndexFlatL2":
            index = faiss.IndexFlatL2(self._dims)
        elif self._index == "IndexFlatIP":
            index = faiss.IndexFlatIP(self._dims)
        elif self._index == "IndexLSH":
            index = faiss.IndexLSH(self._dims, self._lsh_n_bits)

        index.add(vectors)
        faiss.write_index(index, os.path.join(folder_path, "index.nns"))

        names_dict = dict(zip(range(len(names)), names))

        config = self._create_config(names_dict)
        with open(os.path.join(folder_path, "config.json"), 'w') as fp:
            json.dump(config, fp)

    def _create_config(self, names_dict):
        """
        Generated needed config to be able to load and query the index.
        """
        return json.dumps({"model": self.__str__(),
                           "index_file": "index.nns",
                           "faiss_index": self._index,
                           "faiss_lsh_n_bits": self._lsh_n_bits,
                           "dims": self._dims,
                           "names_dict": names_dict})

    def _load(self, index_file_path):
        self.index = faiss.read_index(index_file_path)
        return self

    def _find_near_neighbors(self, vectors, number_of_neighbors=5):
        """
        Bulk lookup the vectors.
        Return only indices (I) list, ignore the distances (D) for now.
        """
        D, I = self.index.search(vectors, number_of_neighbors)
        return I

    def __str__(self):
        return 'faiss'
