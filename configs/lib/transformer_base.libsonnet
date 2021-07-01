local tokenizer = {"type": "sentence_piece", "model_path": std.extVar("SPM_PATH")};

local model_size = 512;
local num_layers = 6;
local embedding_dropout = 0.1;
local encoder_dropout = 0.1;
local decoder_dropout = 0.1;

{
    "type": "placeholder_seq2seq",
    "source_embedder": {
        "type": "basic",
        "token_embedders": {
            "tokens": {
                "type": "embedding_layer",
                "embedding_dim": model_size,
                "vocab_namespace": "tokens",
                "trainable": true,
                "scale_by_embedding_dim": true
            }
        },
    },
    "target_embedder": {
        "type": "basic",
        "token_embedders": {
            "tokens": {
                "type": "embedding_layer",
                "embedding_dim": model_size,
                "vocab_namespace": "tokens",
                "trainable": true,
                "scale_by_embedding_dim": true
            }
        },
    },
    "source_vocab_namespace": "tokens",
    "target_vocab_namespace": "tokens",
    "encoder": {
        "type": "transformer_encoder",
        "size": model_size,
        "num_attention_heads": 8,
        "feedforward_hidden_dim": model_size * 4,
        "num_layers": num_layers,
        "dropout": encoder_dropout,
    },
    "decoder": {
        "type": "transformer_decoder",
        "size": model_size,
        "num_attention_heads": 8,
        "feedforward_hidden_dim": model_size * 4,
        "num_layers": num_layers,
        "dropout": decoder_dropout,
    },
    "token_metrics": {"bleu": {"type": "sacre_bleu", "detokenizer": tokenizer, "lowercase": false},
                      "bp": {"type": "sacre_bleu", "detokenizer": tokenizer, "lowercase": false, "get_bp": true}},
    "share_embeddings": true,
    "share_target_weights": true,
    "max_decoding_length": 100,
    "sequence_loss": {"label_smoothing": 0.1},
    "beam_size": null,
    "initializer": {
        "regexes": [
            ["encoder.*", {"type": "xavier_uniform"}],
            ["decoder.*", {"type": "xavier_uniform"}],
            [".*embed.*", {"type": "normal", "mean": 0, "std": 1 / std.sqrt(model_size)}]
        ],
        "prevent_regexes": [".*norm.*", ".*bias.*"]
    }
}