local seed = std.parseInt(std.extVar("SEED"));
local batch_size = std.parseInt(std.extVar("BATCH_SIZE"));


local factor = 1.0;
local warmup_steps = 8000;


local data = import "data.libsonnet";

local tokenizer = {"type": "sentence_piece", "model_path": std.extVar("SPM_PATH"),
                   "tokenize_with_offsets": true};
local token_indexer = {"tokens": {"type": "single_id", "namespace": "tokens",
                           "start_tokens": ["@start@"], "end_tokens": ["@end@"]}};


local filterers = [
        {"type": "length", "field_name": "source_tokens", "max_length": 100, "min_length": 5},
        {"type": "length", "field_name": "target_tokens", "max_length": 100, "min_length": 5}
];

local reader = {
    "type": "placeholder_seq2seq",
    "file_parser": {"type": "aspec_je"},
    "src_tokenizer": tokenizer,
    "tgt_tokenizer": tokenizer,
    "src_token_indexers": token_indexer,
    "tgt_token_indexers": token_indexer
};


local vocab_path = std.extVar("VOCAB_PATH");
local vocabulary =
    if vocab_path == "null" then {"type": "from_instances",
                                  "tokens_to_add": {"tokens": data["user_defined_symbols"],
                                                    "characters": ["@start@", "[VERB]", "[NOUN]", "@end@"]}
                                  }
    else {"type": "from_files", "directory": std.extVar("VOCAB_PATH")};


{
    "train_data_path": data["train_data_path"],
    "validation_data_path": data["validation_data_path"],
    "test_data_path": data["test_data_path"],
    "dataset_reader": reader + {"filterers": filterers},
    "validation_dataset_reader": reader,
    "vocabulary": vocabulary,
    "data_loader": {
        "batch_sampler": {"type": "max_tokens_sampler",
                          "max_tokens": batch_size,
                          "padding_noise": 0.3,
                          "sorting_keys": ["target_tokens", "source_tokens"]
                          },
        "num_workers": 0,
        "max_instances_in_memory": 1500000
    },
    "validation_data_loader": {
        "batch_sampler": {"type": "max_tokens_sampler",
                          "max_tokens": batch_size,
                          "padding_noise": 0.0}
    },
    "trainer": {
        "num_epochs": 100,
        "patience": 3,
        "cuda_device": -1,
        "grad_norm": 0.25
        ,
        "checkpointer": {
            "num_serialized_models_to_keep": 0
        },
        "validation_metric": "-loss",
        "optimizer": {
            "type": "adam",
            "lr": 0,
            "weight_decay": 1e-07
        },
        "learning_rate_scheduler": {
            "type": "noam",
            "model_size": 512,
            "warmup_steps": warmup_steps,
            "factor": factor
        },
        "callbacks": [{"type": "log_tokens",
                     "input_name_spaces": {"source_tokens": "tokens", "target_tokens": "tokens"},
                     "output_name_spaces": {"prediction_tensor": "tokens"}}],
    },
    "random_seed": seed,
    "numpy_seed": seed,
    "pytorch_seed": seed
}
