# -*- coding: utf-8 -*-


class NearestNeighborSearchAlgorithm:
    def __new__(cls, *args, **kwargs):
        """Determine based on the arguments the appropriate algorithm"""
        algorithm = kwargs.get("algorithm")
        if algorithm == "annoy":
            from algorithms.annoy import Annoy  # noqa

            for i in cls.__subclasses__():
                if i.__name__ == "Annoy":
                    return super().__new__(i)
        elif algorithm == "faiss":
            from algorithms.faiss import Faiss  # noqa

            for i in cls.__subclasses__():
                if i.__name__ == "Faiss":
                    return super().__new__(i)
        else:
            raise NotImplementedError(f"Algorithm '{algorithm}' is not available")

    def __str__(self):
        return self.name

    def load(self, index_file_path):
        pass
