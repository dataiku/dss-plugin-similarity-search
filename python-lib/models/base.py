# -*- coding: utf-8 -*-


class NearestNeighborSearchModel:
    def __new__(cls, *args, **kwargs):
        """Determine based on the arguments the appropriate model"""
        model = kwargs.get("model")
        if model == "annoy":
            from models.annoy import Annoy  # noqa

            for i in cls.__subclasses__():
                if i.__name__ == "Annoy":
                    return super().__new__(i)
        elif model == "faiss":
            from models.faiss import Faiss  # noqa

            for i in cls.__subclasses__():
                if i.__name__ == "Faiss":
                    return super().__new__(i)
        else:
            raise NotImplementedError(f"Similarity search model '{model}' is not available")

    def __str__(self):
        return self.name

    def load(self, index_file_path):
        pass
