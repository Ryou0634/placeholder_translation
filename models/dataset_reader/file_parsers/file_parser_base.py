from allennlp.common import Registrable


# Parser that reads one sentence from one single file.
class FileParser(Registrable):

    def __init__(self, encoding: str = 'utf-8'):
        self.encoding = encoding

    def __call__(self, file_path: str):
        raise NotImplementedError
