local data = import "./data.libsonnet";


local reader = {
    "type": "my_seq2seq",
    "file_parser": {"type": "aspec_je"},
    "src_tokenizer": {"type": "mecab"},
    "tgt_tokenizer": {"type": "spacy"}
  };

{
  "dataset_reader": {
        "type": "interleaving",
        "readers": {
            "train": reader,
            "validation": reader,
            "test": reader,
            "devtest": reader,
        },
        "scheme": "all_at_once"
  },
  "train_data_path": std.toString({
        "train": data["train_data_path"],
        "validation": data["validation_data_path"],
        "test": data["test_data_path"],
        "devtest": data["devtest_data_path"],
  })
}