local aspec_je_data_dir = std.extVar("ASPEC_JE_DIR");


{
    "train_data_path": aspec_je_data_dir + "/train",
    "validation_data_path": aspec_je_data_dir + "/dev",
    "test_data_path": aspec_je_data_dir + "/test",
    "devtest_data_path": aspec_je_data_dir + "/devtest",
    "file_parser": {"type": "aspec_je"},
    "user_defined_symbols": ["[PLACEHOLDER]", "@start@", "@end@", "@sep@", "[NOUN]", "[VERB]",
                             "<NNS>", "<NN>", "<VB>", "<VBD>", "<VBG>", "<VBN>", "<VBP>", "<VBZ>"]
}
