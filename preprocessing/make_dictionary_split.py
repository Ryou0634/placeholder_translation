import tqdm
import itertools
from pathlib import Path
import click
import sys

sys.path.append("./")

from models.dataset_reader.file_parsers.parallel_file_parsers import AspecJEParser
from models.placeholder.word_replacer import WordReplacer, dump_dictionary


@click.command()
@click.argument("dictionary_dir", type=click.Path(exists=True))
@click.argument("aspec-path", type=click.Path(exists=True))
def make_dictionary_split(dictionary_dir, aspec_path):
    dictionary_dir = Path(dictionary_dir)
    aspec_path = Path(aspec_path)
    for dict_file in ["word-dictionary-noun.tsv", "word-dictionary-verb-filtered.tsv"]:
        dict_path = dictionary_dir / dict_file

        parser = AspecJEParser()
        train_replacer = WordReplacer(dict_path)
        dev_test_replacer = WordReplacer(dict_path)

        train_entries = set()
        for i, (ja, en) in tqdm.tqdm(enumerate(parser(aspec_path / "train"))):
            entries = train_replacer.find_entries(ja, en)
            train_entries.update(entries)

        dev_test_entries = set()
        for ja, en in tqdm.tqdm(
            itertools.chain(parser(aspec_path / "dev"), parser(aspec_path / "test"), parser(aspec_path / "devtest"))
        ):
            entries = dev_test_replacer.find_entries(ja, en)
            dev_test_entries.update(entries)

        dev_test_entry_lemmas = {e.target_lemma for e in dev_test_entries}
        dev_test_seen_lemmas = set()
        dev_test_unseen_lemmas = set()
        for i, target_lemma in enumerate(dev_test_entry_lemmas):
            if i % 2:
                dev_test_seen_lemmas.add(target_lemma)
            else:
                dev_test_unseen_lemmas.add(target_lemma)

        train_entries = [e for e in train_entries if e.target_lemma not in dev_test_unseen_lemmas]

        dev_test_unseen_entries = [e for e in dev_test_entries if e.target_lemma in dev_test_unseen_lemmas]
        dev_test_seen_entries = [e for e in dev_test_entries if e.target_lemma in dev_test_seen_lemmas]

        stem = Path(dict_file).stem.replace("-filtered", "")
        dump_dictionary(train_entries, dictionary_dir / f"{stem}.train.tsv")
        dump_dictionary(dev_test_unseen_entries, dictionary_dir / f"{stem}.devtest-unseen.tsv")
        dump_dictionary(dev_test_seen_entries, dictionary_dir / f"{stem}.devtest-seen.tsv")


if __name__ == "__main__":
    make_dictionary_split()
