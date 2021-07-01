local base = import "lib/base.libsonnet";
local model = import "lib/transformer_base.libsonnet";


local placeholder = {
    "word_replacers": [{"type": "code-switching",
                        "dictionary_path": std.extVar("NOUN_DICT_PATH"),
                        "max_count": 20,
                        "replace_random": true,
                        "use_lemma": true},
                        {"type": "code-switching",
                        "dictionary_path": std.extVar("VERB_DICT_PATH"),
                        "max_count": 2000,
                        "replace_random": true,
                        "use_lemma": true},
                        ],
    "replace_ratio": 1.0,
    "mode": "augment"
};

local vaildation_placeholder = {
    "word_replacers": [{"type": "code-switching",
                       "dictionary_path": std.extVar("VALIDATION_DICT_PATH"),
                       "max_count": 20,
                       "replace_random": false,
                       "use_lemma": true}],
    "replace_ratio": 1.0,
    "mode": "replace"
};

base + {
    "model": model,
    "dataset_reader": base["dataset_reader"] + placeholder,
    "validation_dataset_reader": base["validation_dataset_reader"] + vaildation_placeholder,
}

