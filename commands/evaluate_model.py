from pathlib import Path
import json
import os
import click


@click.command()
@click.argument("serialization-dir", type=click.Path(exists=True))
@click.argument("dictionary-dir", type=str)
@click.argument("test-data-path", type=str)
@click.option("--cuda_device", type=int, default=-1)
@click.option("--replace_ratio", type=float, default=1.0)
@click.option("--specified_inflection", type=str)
@click.option("--evaluate_lemma", is_flag=True)
@click.option("--keep_placeholder", is_flag=True)
@click.option("--only_placeholder", is_flag=True)
@click.option("--dummy", is_flag=True)
@click.option("--translation", is_flag=True)
def evaluate_model(
    serialization_dir: str,
    dictionary_dir: str,
    test_data_path: str,
    cuda_device: int,
    replace_ratio: float,
    specified_inflection: str,
    evaluate_lemma: bool,
    keep_placeholder: bool,
    only_placeholder: bool,
    dummy: bool,
    translation: bool,
):
    serialization_dir = Path(serialization_dir)
    dictionary_dir = Path(dictionary_dir)
    stem = Path(test_data_path).stem

    config = json.load(open(serialization_dir / "config.json", "r"))
    replacer_type = config["validation_dataset_reader"]["word_replacers"][0]["type"]
    if dummy:
        replacer_type = "dummy"

    for pos, placeholder_token, max_count in [("noun", "[NOUN]", 20), ("verb", "[VERB]", 2000)]:
        for split in ["seen", "unseen"]:
            dictionary_name = f"word-dictionary-{pos}.devtest-{split}.tsv"

            override = {
                "validation_dataset_reader": {
                    "word_replacers": [
                        {
                            "type": replacer_type,
                            "dictionary_path": str(dictionary_dir / dictionary_name),
                            "max_count": max_count,
                            "replace_random": False,
                        }
                    ],
                    "replace_ratio": replace_ratio,
                }
            }
            if "placeholder" in replacer_type:
                override["validation_dataset_reader"]["word_replacers"][0]["placeholder_token"] = placeholder_token

            output_file = serialization_dir / f"{stem}_{pos}_{split}.json"

            if evaluate_lemma:
                override["model"] = {"evaluate_lemma": True}
                output_file = serialization_dir / f"{stem}_{pos}_{split}_evaluate_lemma.json"

            if specified_inflection:
                override["model"] = {"specified_inflection": specified_inflection}
                output_file = serialization_dir / f"{stem}_{pos}_{split}_with_{specified_inflection}.json"

            if keep_placeholder:
                override["model"] = {"keep_placeholder": True}
                output_file = serialization_dir / f"{stem}_{pos}_{split}_keep_placeholder.json"

            if only_placeholder:
                override["validation_dataset_reader"]["mode"] = "only_placeholder"
                output_file = serialization_dir / f"{stem}_{pos}_{split}_only_placeholder.json"

            if translation:
                dir_name = serialization_dir.name
                if specified_inflection:
                    dir_name += f"_with_{specified_inflection}"

                translation_save_dir = serialization_dir / "translation" / output_file.stem
                cmd = (
                    f"python cli.py translate {serialization_dir} "
                    f"--include-package models "
                    f"--cuda-device {cuda_device} --overrides '{json.dumps(override)}' "
                    f"--save-directory {translation_save_dir}"
                )

                print(cmd)
                os.system(cmd)
            else:
                cmd = (
                    f"allennlp evaluate {serialization_dir} {test_data_path} "
                    f"--include-package models "
                    f"-o '{json.dumps(override)}' --cuda-device {cuda_device} --output-file {output_file}"
                )
                print(cmd)
                os.system(cmd)


if __name__ == "__main__":

    evaluate_model()
