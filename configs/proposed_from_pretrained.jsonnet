local base = import "lib/base.libsonnet";
local model = import "lib/transformer_base.libsonnet";
local pretrained_weight = std.extVar("PRETRAINED_WEIGHT");

local model_size = 512;
local num_layers = 2;

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
    "mode": "only_placeholder",
    "use_character": true
};

local vaildation_placeholder = {
    "word_replacers": [{"type": "placeholder",
                       "dictionary_path": std.extVar("VALIDATION_DICT_PATH"),
                       "max_count": 30,
                       "placeholder_token": "[NOUN]",
                       "replace_random": false}],
    "replace_ratio": 1.0,
    "mode": "replace",
    "use_character": true
};

base + {
    "model": model + {"type": "inflectional_placeholder_seq2seq",
                        "inflection_character_decoder": {
                            "type": "transformer_decoder",
                            "size": model_size,
                            "num_attention_heads": 8,
                            "feedforward_hidden_dim": model_size * 2,
                            "num_layers": num_layers,
                            "dropout": 0.1,
                        },
                      "max_inflection_length": 30,
                      "only_train_character_decoder": true,
                      "initializer": {"regexes": [
                            [".*", {"type": "pretrained",
                                    "weights_file_path": pretrained_weight}]
                                    ]}
                      },
    "dataset_reader": base["dataset_reader"] + placeholder,
    "validation_dataset_reader": base["validation_dataset_reader"] + vaildation_placeholder,
    "trainer": base["trainer"] + {"optimizer": {"parameter_groups": [
        [
            ["^encoder"],
            {"requires_grad": false}
        ],
        [
            ["^decoder.*"],
            {"requires_grad": false}
        ],
        [
            ["^output_projection_layer.*"],
            {"requires_grad": false}
        ],
        [
            ["^source_embedder.*"],
            {"requires_grad": false}
        ],
        [
            ["^target_embedder.*"],
            {"requires_grad": false}
        ],
        [
            ["^inflection_generator.*"],
            {"requires_grad": true}
        ]
    ]},
    "validation_metric": "-loss",
    "patience": 5,
    }
}

