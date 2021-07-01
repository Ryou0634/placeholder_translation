from typing import List, Dict
from pathlib import Path
import click
import logging

import torch

from allennlp.nn.util import move_to_device
from allennlp.models.archival import load_archive
from allennlp.common.util import import_module_and_submodules, prepare_environment
from allennlp.data import DatasetReader
from allennlp.models.model import Model
from allennlp.data.data_loaders import MultiProcessDataLoader

RESULT_DIR = Path("results")

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def generate_translation(
    input_file: str,
    model: Model,
    dataset_reader: DatasetReader,
    source_arg_name: str = "source_tokens",
    target_arg_name: str = "target_tokens",
    cuda_device: int = -1,
):

    data_loader = MultiProcessDataLoader(reader=dataset_reader, data_path=input_file, batch_size=32)
    data_loader.index_with(model.vocab)

    if cuda_device == -1:
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{cuda_device}")
    model = model.to(device)

    text_dict: Dict[str, List[str]] = {"source": [], "target": [], "prediction": []}
    print("Processing source and target texts...")
    for instance in data_loader.iter_instances():
        source_tokens = [t.text for t in instance[source_arg_name]]
        src_tokenized = " ".join(source_tokens)
        text_dict["source"].append(src_tokenized)

        target_tokens = [t.text for t in instance[target_arg_name]]
        tgt_tokenized = " ".join(target_tokens)
        text_dict["target"].append(tgt_tokenized)

    print("Translating...")
    for batch in data_loader:
        batch = move_to_device(batch, device=device)
        output_dict = model.forward(**batch)
        prediction_tensor_list = output_dict["prediction"]
        for prediction_tensor in prediction_tensor_list:
            tokenized = " ".join(prediction_tensor)
            text_dict["prediction"].append(tokenized)
    return text_dict


@click.command()
@click.argument("archive_directory", type=str)
@click.option("--source-arg-name", type=str, default="source_tokens")
@click.option("--target-arg-name", type=str, default="target_tokens")
@click.option("--input-file", type=str)
@click.option("--weights-file", type=str)
@click.option("--save-directory", type=str)
@click.option("--cuda-device", type=int, default=-1)
@click.option("--overrides", type=str)
@click.option("--include-package", type=str, multiple=True)
def translate(
    archive_directory: str,
    source_arg_name: str,
    target_arg_name: str,
    input_file: str,
    weights_file: str,
    save_directory: str,
    cuda_device: int,
    overrides: str,
    include_package: List[str],
):
    for package_name in include_package:
        import_module_and_submodules(package_name)
    archive_directory = Path(archive_directory)
    archive = load_archive(str(archive_directory / "model.tar.gz"), cuda_device, overrides, weights_file)
    config = archive.config

    prepare_environment(config)

    model = archive.model
    model.eval()

    input_file = input_file or config["test_data_path"]

    validation_dataset_reader_params = config.pop("validation_dataset_reader", None)
    if validation_dataset_reader_params is not None:
        dataset_reader = DatasetReader.from_params(validation_dataset_reader_params)
    else:
        dataset_reader = DatasetReader.from_params(config.pop("dataset_reader"))

    text_dict = generate_translation(input_file, model, dataset_reader, source_arg_name, target_arg_name, cuda_device)

    input_file_name = Path(input_file).name

    if save_directory:
        save_directory = Path(save_directory)
    else:
        save_directory = archive_directory / "translations"
    save_directory.mkdir(exist_ok=True, parents=True)
    for file_suffix, texts in text_dict.items():
        save_path = str(save_directory / f"{input_file_name}.{file_suffix}.txt")
        print(f"Dumping the files to {save_directory}...")
        with open(save_path, "w") as f:
            f.write("\n".join(texts))


if __name__ == "__main__":
    translate()
