import spacy
import tqdm
import click
from pathlib import Path
import sys

sys.path.append("./")
from models.placeholder.word_replacer import Entry, dump_dictionary, parse_dictionary


AUX_VERBS = ["be", "is", "are", "was", "were", "has been", "have been", "has", "have"]


@click.command()
@click.argument("verb-dictionary-path", type=click.Path(exists=True))
@click.argument("output-dir", type=click.Path(exists=True))
def filter_verb_dictionary(verb_dictionary_path: str, output_dir: str):
    nlp = spacy.load("en_core_web_sm")

    filtered_entries = []
    for entry in tqdm.tqdm(parse_dictionary(verb_dictionary_path)):
        ja, _, ja_lemma, en, _, en_lemma, count = entry
        if "を" in ja or "ついて" in ja or "に" in ja or "が" in ja or "で" in ja:
            continue
        en_tokens = [t for t in nlp(en)]

        if en_tokens[0].tag_ in {"VBN", "VBD"} or en_tokens[0].text in {"has", "have", "had"}:
            if ja[-1] in {"た", "だ"}:
                filtered_entries.append(entry)
            elif ja[-1] in {"し", "い"}:
                new_entry = Entry(ja + "た", ja, ja_lemma, en, en, en_lemma, count)
                filtered_entries.append(new_entry)
        elif en_tokens[0].tag_ in {"VB", "VBP", "VBZ", "VBG"}:
            if ja == ja_lemma:
                filtered_entries.append(entry)
    filtered_entries = list(set(filtered_entries))

    suru_inflections = sorted(
        [
            "する",
            "した",
            "される",
            "された",
            "して",
            "していた",
            "されてきた",
            "してきた",
            "してきている",
            "され",
            "している",
            "されている",
            "されていた",
            "させる",
            "させた",
            "させていた",
            "させている",
            "させて",
            "されて",
            "させていて",
        ],
        key=lambda x: -len(x),
    )

    cleaned_entries = []
    for entry in tqdm.tqdm(filtered_entries):
        if not any([entry.source_matching_pattern.endswith(suffix) for suffix in suru_inflections]):
            continue

        if entry.source_matching_pattern.startswith("し") or entry.source_matching_pattern.startswith("する"):
            continue

        ja_lemma = entry.source_lemma
        for suffix in suru_inflections:
            ja_lemma = ja_lemma.replace(suffix, "")

        new_entry = Entry(
            entry.source_matching_pattern,
            entry.source_replace_pattern,
            ja_lemma,
            entry.target_matching_pattern,
            entry.target_replace_pattern,
            entry.target_lemma,
            entry.count,
        )
        cleaned_entries.append(new_entry)

    augmented_entries = []
    for entry in tqdm.tqdm(cleaned_entries):
        source, _, source_lemma, target, _, target_lemma, count = entry
        en_tokens = entry.target_matching_pattern.split()

        if en_tokens[0] in {"is", "are"}:
            for aux in ["is", "are"]:
                augmented_target = " ".join([aux] + en_tokens[1:])
                augmented_entries.append(
                    Entry(source, source, source_lemma, augmented_target, augmented_target, target_lemma, count)
                )

        if en_tokens[0] in {"was", "were"}:
            for aux in ["was", "were"]:
                augmented_target = " ".join([aux] + en_tokens[1:])
                augmented_entries.append(
                    Entry(source, source, source_lemma, augmented_target, augmented_target, target_lemma, count)
                )

        if en_tokens[0] in {"has", "have"}:
            for aux in ["has", "have"]:
                augmented_target = " ".join([aux] + en_tokens[1:])
                augmented_entries.append(
                    Entry(source, source, source_lemma, augmented_target, augmented_target, target_lemma, count)
                )

    final_entries = []
    for entry in list(set(augmented_entries + cleaned_entries)):
        target_words = entry.target_matching_pattern.split()
        for w in AUX_VERBS:
            if w in target_words:
                target_words.remove(w)

        target_replace_pattern = " ".join(target_words)
        if len(target_replace_pattern.strip()) == 0:
            continue
        final_entries.append(
            Entry(
                entry.source_matching_pattern,
                entry.source_lemma,
                entry.source_lemma,
                entry.target_matching_pattern,
                target_replace_pattern,
                entry.target_lemma,
                entry.count,
            )
        )

    dump_dictionary(final_entries, Path(output_dir) / "word-dictionary-verb-filtered.tsv")


if __name__ == "__main__":
    filter_verb_dictionary()
