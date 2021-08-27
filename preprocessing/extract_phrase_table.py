import tqdm
from collections import Counter
import logging
import click

import re


def read_giza_output(path: str):
    """
    Parameters
    ----------
    path : The path to alignment file. (**.AA3.final)
    """
    index_pattern = r"\(\{ [\d* ]*\}\)"
    index_pattern_for_split = " " + index_pattern + " "
    index_pattern = re.compile(index_pattern)
    index_pattern_for_split = re.compile(index_pattern_for_split)
    digit_pattern = re.compile(r"\d+")

    instance = {}
    with open(path, "r") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if i % 3 == 0:
                instance["alignment_score"] = float(line.split(" : ")[1])
            elif i % 3 == 1:
                instance["target_tokens"] = line.split()
            elif i % 3 == 2:
                aligned_source = line + " "
                instance["source_tokens"] = re.split(index_pattern_for_split, aligned_source)[:-1]

                instance["src2tgt_alignment"] = [
                    tuple(map(lambda x: x - 1, map(int, re.findall(digit_pattern, txt))))
                    for txt in re.findall(index_pattern, aligned_source)
                ]
                assert len(instance["src2tgt_alignment"]) == len(instance["source_tokens"])
                yield instance
                instance = {}


def extract_phrase_pairs(instance, max_phrase_length: int = 6):
    def to_extract(src_start, src_end, tgt_start, tgt_end, src2tgt_alignment):
        if tgt_end == 0:
            return False

        for t in range(tgt_start, tgt_end + 1, 1):
            is_aligned_to_src = [t in src2tgt_alignment[s] for s in range(src_start, src_end + 1, 1)]
            if not any(is_aligned_to_src):
                return False
        return True

    phrase_pairs = []
    source_length = len(instance["source_tokens"])
    for src_start in range(1, source_length):  # the first token is NULL token
        max_src_end = min(source_length, src_start + max_phrase_length)
        for src_end in range(src_start, max_src_end):
            tgt_start, tgt_end = len(instance["target_tokens"]), 0
            for src_idx, tgt_indices in enumerate(instance["src2tgt_alignment"]):
                for tgt_idx in tgt_indices:
                    if src_start <= src_idx <= src_end:
                        tgt_start = min(tgt_idx, tgt_start)
                        tgt_end = max(tgt_idx, tgt_end)
            if tgt_end - tgt_start >= max_phrase_length:
                continue
            if to_extract(src_start, src_end, tgt_start, tgt_end, instance["src2tgt_alignment"]):
                src_phrase = " ".join(instance["source_tokens"][src_start : src_end + 1])
                tgt_phrase = " ".join(instance["target_tokens"][tgt_start : tgt_end + 1])
                phrase_pairs.append((src_phrase, tgt_phrase))
    return phrase_pairs


logger = logging.getLogger(__name__)


@click.command()
@click.argument("alignment-file-path", type=click.Path(exists=True))
@click.argument("output-path", type=click.Path(exists=False))
def extract_phrase_table(alignment_file_path: str, output_path: str):
    counter = Counter()
    logging.info(f"Reading data from {alignment_file_path}...")
    for instance in tqdm.tqdm(read_giza_output(alignment_file_path)):
        phrase_pairs = extract_phrase_pairs(instance)
        for src_phrase, tgt_phrase in phrase_pairs:
            counter[(src_phrase, tgt_phrase)] += 1
    logging.info(f"Extracted phrases: {len(counter)}")
    with open(output_path, "w") as f:
        for (src_phrase, tgt_phrase), count in counter.most_common():
            f.write(f"{src_phrase}\t{tgt_phrase}\t{count}\n")


if __name__ == "__main__":
    extract_phrase_table()
