from collections import Counter
import spacy
import tqdm
from pathlib import Path
import click

import sys

sys.path.append("./")

from models.tokenizers.mecab_tokenizer import MeCabTokenizer


def remove_marks(text: str):
    text = text.replace("(", "")
    text = text.replace(")", "")
    text = text.replace(",", "")
    text = text.strip()
    return text


def make_noun_dictionary(output_dir: str, phrase_table: Counter):

    ja_tokenizer = MeCabTokenizer(with_pos_tag=True)
    nlp = spacy.load("en_core_web_sm")

    noun_counter = Counter()
    for (ja_word, en_word), count in tqdm.tqdm(phrase_table.items()):
        ja_word = remove_marks(ja_word)

        en_token = nlp(en_word)
        en_pos_list = [e.pos_ for e in en_token]
        ja_token = ja_tokenizer.tokenize(ja_word)
        ja_pos_list = [j.pos_ for j in ja_token]
        ja_tag_list = [j.tag_ for j in ja_token]

        if (
            en_pos_list[0] == "NOUN"
            and ja_pos_list[0] == "名詞"
            and len(set(en_pos_list)) == 1
            and len(set(ja_pos_list)) == 1
        ):
            if "数" in ja_tag_list or "接尾" in ja_tag_list:
                continue
            lemma_en_word = " ".join([e.lemma_ if i == len(en_token) - 1 else e.text for i, e in enumerate(en_token)])
            ja_word = "".join(ja_word.split())

            # handles case like word: "UN" -> lemma: "un"
            if en_word.lower() == lemma_en_word:
                lemma_en_word = en_word

            if len(ja_word) <= 1:
                continue

            noun_counter[(ja_word, ja_word, en_word, lemma_en_word)] += count

    with open(Path(output_dir) / "word-dictionary-noun.tsv", "w") as f:
        for (j, j_lemma, e, e_lemma), c in noun_counter.most_common():
            f.write(f"{j}\t{j}\t{j_lemma}\t{e}\t{e}\t{e_lemma}\t{c}\n")


def make_verb_dictionary(output_dir: str, phrase_table: Counter):
    ja_tokenizer = MeCabTokenizer(with_pos_tag=True)
    nlp = spacy.load("en_core_web_sm")

    verb_counter = Counter()
    for (ja_word, en_word), count in tqdm.tqdm(phrase_table.items()):
        ja_word = remove_marks(ja_word)

        en_tokens = [t for t in nlp(en_word)]
        en_pos = [t.pos_ for t in en_tokens]
        en_text = [t.text for t in en_tokens]
        if en_text == ["can"]:
            continue

        if not (en_pos == ["VERB"] or en_pos == ["AUX", "VERB"] or en_pos == ["AUX", "AUX", "VERB"]):
            continue

        ja_tokens = ja_tokenizer.tokenize(ja_word)
        while len(ja_tokens) > 0 and ja_tokens[0].pos_ not in {"動詞", "名詞", "助動詞"}:
            ja_tokens = ja_tokens[1:]
        if len(ja_tokens) == 0:
            continue

        ja_lemma = [t.lemma_ for t in ja_tokens]

        if not any([l == "する" for l in ja_lemma]):
            continue

        ja_word = "".join([j.text for j in ja_tokens])

        lemma_tokens = ja_tokens
        while lemma_tokens[-1].pos_ == "助動詞":
            lemma_tokens = lemma_tokens[:-1]
        lemma_ja_word = "".join(
            [j.lemma_ if i == len(lemma_tokens) - 1 else j.text for i, j in enumerate(lemma_tokens)]
        )

        lemma_en_word = en_tokens[-1].lemma_
        en_word = " ".join([t.text for t in en_tokens])

        # handles case like word: "UN" -> lemma: "un"
        if en_word.lower() == lemma_en_word:
            lemma_en_word = en_word

        if len(ja_word) <= 1:
            continue
        verb_counter[(ja_word, lemma_ja_word, en_word, lemma_en_word)] += count

    with open(Path(output_dir) / "word-dictionary-verb.tsv", "w") as f:
        for (j, j_lemma, e, e_lemma), c in verb_counter.most_common():
            f.write(f"{j}\t{j}\t{j_lemma}\t{e}\t{e}\t{e_lemma}\t{c}\n")


@click.command()
@click.argument("phrase-table-path", type=click.Path(exists=True))
@click.argument("output-dir", type=click.Path(exists=True))
@click.argument("min_count", type=int, default=10)
@click.option("--noun-only", is_flag=True)
@click.option("--verb-only", is_flag=True)
def make_dictionary(phrase_table_path: str, output_dir: str, min_count: int, noun_only: bool, verb_only: bool):
    assert not (noun_only & verb_only)

    phrase_table = Counter()
    with open(phrase_table_path, "r") as f:
        for line in f:
            src, tgt, count = line.split("\t")

            count = int(count.strip())
            phrase_table[(src, tgt)] = count
            if count < min_count:
                break

    if not verb_only:
        make_noun_dictionary(output_dir, phrase_table)

    if not noun_only:
        make_verb_dictionary(output_dir, phrase_table)


if __name__ == "__main__":
    make_dictionary()
