# Placeholder Translation
This repository contains the source code to train machine translation systems, with a focus on placeholder translation.
The code is based on [allennlp](https://github.com/allenai/allennlp).

You can use this code to reproduce the results in the paper ["Modeling Target-side Inflection in Placeholder Translation"](http://arxiv.org/abs/2107.00334).


## Install Dependencies
Please use python 3.8.5.  
Install [Poetry](https://python-poetry.org/) and hit the following command.

```bash
poetry install
poetry run python -m spacy download en_core_web_sm
```

## Preparing Dataset
To reproduce the results on ASPEC-JE, get the corpus from [here](https://jipsti.jst.go.jp/aspec/).

## Running Experiments

### Preprocessing
```bash
# in this example, we save results under 'results', but you can change this as you want.

# Train sentencepiece
export ASPEC_JE_DIR=/PATH_TO_CORPUS/ASPEC-JE; # specify where you place the ASPEC Corpus
poetry run python cli.py train-spm-from-config configs/lib/data.libsonnet results/sentencepiece/aspec_je_16k 16000 --include-package models --concat-parallel

# dry-run to make vocabulary files
export VOCAB_PATH=null;
export SPM_PATH=results/sentencepiece/aspec_je_16k.model;

# the following variables don't matter at this step, but need to be defined anyway.
export SEED=0;
export BATCH_SIZE=2048;
export NOUN_DICT_PATH=data/ASPEC-JE-word-dictionary/word-dictionary-noun.train.tsv;
export VERB_DICT_PATH=data/ASPEC-JE-word-dictionary/word-dictionary-verb.train.tsv;
export VALIDATION_DICT_PATH=data/ASPEC-JE-word-dictionary/word-dictionary-noun.devtest-seen.tsv;

poetry run allennlp train configs/proposed.jsonnet --file-friendly-logging --include-package models --s results/dry-run --dry-run
```

### Reproduce experiments in the paper

#### Training
```bash
export ASPEC_JE_DIR=/PATH_TO_CORPUS/ASPEC-JE;
export SPM_PATH=results/sentencepiece/aspec_je_16k.model;
export VOCAB_PATH=results/dry-run/vocabulary;
export NOUN_DICT_PATH=data/ASPEC-JE-word-dictionary/word-dictionary-noun.train.tsv;
export VERB_DICT_PATH=data/ASPEC-JE-word-dictionary/word-dictionary-verb.train.tsv;
export VALIDATION_DICT_PATH=data/ASPEC-JE-word-dictionary/word-dictionary-noun.devtest-seen.tsv;


export SEED=0;
export BATCH_SIZE=2048;

# baseline
poetry run allennlp train configs/baseline.jsonnet --file-friendly-logging --include-package models --s results/baseline -o '{"trainer": {"cuda_device": 0, "use_amp": true}}' -f

# PH
poetry run allennlp train configs/placeholder.jsonnet --file-friendly-logging --include-package models --s results/placeolder -o '{"trainer": {"cuda_device": 0, "use_amp": true}}' -f

# CS
poetry run allennlp train configs/code-switching.jsonnet --file-friendly-logging --include-package models --s results/code-switching -o '{"trainer": {"cuda_device": 0, "use_amp": true}}' -f

# CS (lemma)
poetry run allennlp train configs/code-switching.jsonnet --file-friendly-logging --include-package models --s results/code-switching_lamma -o '{"trainer": {"cuda_device": 0, "use_amp": true}}' -f

# PH (morph)
poetry run allennlp train configs/placeholder_morph.jsonnet --file-friendly-logging --include-package models --s results/placeholder_morph -o '{"trainer": {"cuda_device": 0, "use_amp": true}}' -f

# Proposed (fine-tuned from PH)
export PRETRAINED_WEIGHT=results/placeolder/best.th;
poetry run allennlp train configs/proposed_from_pretrained.jsonnet --file-friendly-logging --include-package models --s results/proposed -o '{"trainer": {"cuda_device": 0, "use_amp": true}}' -f
```

#### Evaluation
The following command evaluate the model and save results under the serialization directory.
```bash
poetry run python commands/evaluate_model.py results/baseline data/ASPEC-JE-word-dictionary  /PATH_TO_CORPUS/ASPEC-JE/test --cuda_device 0
```