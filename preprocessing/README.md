# Generating Word Dictionaries
First, you need an output file from [GIZA++](https://sinaahmadi.github.io/posts/sentence-alignment-using-giza.html).

```bash
# run the following commands under the root directory
poetry run python  -m spacy download en_core_web_sm
poetry run python preprocessing/extract_phrase_table.py OUTPUT-FROM-GIZA.AA3.final data/phrase_table.tsv
poetry run python preprocessing/make_dictionary.py data/phrase_table.tsv data
poetry run python preprocessing/filter_verb_dictionary.py data/word-dictionary-verb.tsv data
poetry run python preprocessing/make_dictionary_split.py data PATH-TO-ASPEC/ASPEC-JE
```