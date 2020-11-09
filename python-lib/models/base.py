class BaseNNS(object):

    def __new__(cls, *args, **kwargs):
        """
        Determine based on the arguments the appropriate model.
        """
        model = kwargs.get("model")
        if model == "annoy":
            from models.annoy import Annoy
            for i in cls.__subclasses__():
                if i.__name__ == "Annoy":
                    return super().__new__(i)
        elif model == "faiss":
            from models.faiss import Faiss
            for i in cls.__subclasses__():
                if i.__name__ == "Faiss":
                    return super().__new__(i)
        else:
            raise ValueError("The model for {} doesn't exist!".format(model))

    def _load(self, index_file_path):
        pass

    def __str__(self):
        return self.name
