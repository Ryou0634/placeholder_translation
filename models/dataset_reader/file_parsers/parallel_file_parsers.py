from pathlib import Path
from .file_parser_base import FileParser


# Parser that reads two (parallel) sentences from one single file.
@FileParser.register("aspec_je")
class AspecJEParser(FileParser):
    def __init__(self, read_only_first: bool = True):
        """
        Parameters
        ----------
        read_only_first: bool
            The ASPEC-JE corpus is ordered in the order of quality of the bitext.
            It would degrade performance to include all the data.
        """
        self.read_only_first = read_only_first

    def __call__(self, file_path: str):
        # The train split consists of three files
        # Example: "ASPEC-JE/train" ->
        # ["ASPEC-JE/train/train-1.txt", "ASPEC-JE/train/train-2.txt", "ASPEC-JE/train/train-3.txt"]
        file_path = Path(file_path)
        if file_path.stem == "train":
            file_path_list = [file_path / (file_path.stem + f"-{n}.txt") for n in [1, 2, 3]]
        else:
            file_path_list = [file_path / (file_path.stem + ".txt")]

        if self.read_only_first:
            file_path_list = file_path_list[:1]

        for file_path in file_path_list:
            with open(file_path, "r") as f:
                for line in f:
                    if file_path.parent.stem == "train":
                        score, doc_id, no, ja, en = line.strip().split("|||")
                    else:
                        doc_id, no, ja, en = line.strip().split("|||")
                    yield ja.strip(), en.strip()


@FileParser.register("aspec_jc")
class AspecJCParser(FileParser):
    def __call__(self, file_path: str):
        # file_path is actually the directory. Get the text file with the same name.
        # Example: "ASPEC-JC/dev" -> "ASPEC-JC/dev/dev.txt"
        file_path = Path(file_path) / (Path(file_path).stem + ".txt")
        with open(file_path, "r") as f:
            for line in f:
                doc_id_no, ja, zh = line.strip().split("|||")
                yield ja.strip(), zh.strip()
