from typing import Union, List
import _jsonnet
import json
import itertools
from pathlib import Path
from models.dataset_reader.file_parsers import FileParser
from models.tokenizers.sp_tokenizer import train_sentencepiece_from_iterable
import click
from allennlp.common.util import import_module_and_submodules
from allennlp.common.params import Params, _environment_variables
import sys
import logging

logging.basicConfig(format="%(asctime)s | %(levelname)s : %(message)s", level=logging.INFO, stream=sys.stdout)


def _extract_from_container_generator(generator, idx: int):
    for element in generator:
        yield element[idx]


@click.command()
@click.argument("config_path", type=click.Path(exists=True))
@click.argument("save_path", type=click.Path())
@click.argument("model_size", type=int)
@click.option("--normalization", type=str, default="nmt_nfkc")
@click.option("--include-package", type=str, multiple=True)
@click.option("--concat-parallel", is_flag=True)
@click.option("--src-only", is_flag=True)
@click.option("--tgt-only", is_flag=True)
@click.option("--character-coverage", type=float, default=0.9995)
@click.option("--force", is_flag=True)
def train_spm_from_config(
    config_path: str,
    save_path: str,
    model_size: int,
    normalization: str,
    include_package: Union[List[str], str],
    concat_parallel: bool,
    src_only: bool,
    tgt_only: bool,
    character_coverage: float,
    force: bool,
):

    if not force and Path(save_path + ".model").exists():
        raise Exception(f"File already exists: {save_path}")

    if isinstance(include_package, str):
        include_package = [include_package]
    for i in include_package:
        import_module_and_submodules(i)

    data_config = json.loads(_jsonnet.evaluate_file(config_path, ext_vars=_environment_variables()))
    data_path = data_config["train_data_path"]

    parser = FileParser.from_params(Params(data_config["file_parser"]))
    data_stream = parser(data_path)

    if concat_parallel:
        data_stream = itertools.chain.from_iterable(data_stream)
    elif src_only:
        data_stream = _extract_from_container_generator(data_stream, 0)
    elif tgt_only:
        data_stream = _extract_from_container_generator(data_stream, 1)

    if "user_defined_symbols" in data_config:
        user_defined_symbols = data_config["user_defined_symbols"]
    else:
        user_defined_symbols = None

    train_sentencepiece_from_iterable(
        data_stream,
        save_path,
        model_size,
        character_coverage=character_coverage,
        normalization_rule_name=normalization,
        user_defined_symbols=user_defined_symbols,
    )


if __name__ == "__main__":
    train_spm_from_config()
