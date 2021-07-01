local base = import "lib/base.libsonnet";
local model = import "lib/transformer_base.libsonnet";


local placeholder = {
    "word_replacers": [{"type": "placeholder",
                        "dictionary_path": std.extVar("NOUN_DICT_PATH"),
                        "placeholder_token": "[NOUN]",
                        "max_count": 20,
                        "replace_random": true},
                        {"type": "placeholder",
                        "dictionary_path": std.extVar("VERB_DICT_PATH"),
                        "placeholder_token": "[VERB]",
                        "max_count": 2000,
                        "replace_random": true},
                        ],
    "replace_ratio": 1.0,
    "mode": "augment"
};

local vaildation_placeholder = {
    "word_replacers": [{"type": "placeholder",
                       "dictionary_path": std.extVar("VALIDATION_DICT_PATH"),
                       "max_count": 20,
                       "placeholder_token": "[NOUN]",
                       "replace_random": false}],
    "replace_ratio": 1.0,
    "mode": "replace"
};

base + {
    "model": model,
    "dataset_reader": base["dataset_reader"] + placeholder,
    "validation_dataset_reader": base["validation_dataset_reader"] + vaildation_placeholder,
}

