from chemnlp.data.ner import group_tokens_by_labels, join_punctuation


def test_tokens_by_label():
    tokens = ["a", "b", "c", "d", "e", "f"]
    labels = [0, 1, 1, 0, 1, 0]
    grouped_tokens = group_tokens_by_labels(tokens, labels, join=False)
    assert grouped_tokens == [["b"], ["c"], ["e"]]

    labels = [0, 1, 2, 0, 1, 0]
    grouped_tokens = group_tokens_by_labels(tokens, labels, join=False)
    assert grouped_tokens == [["b", "c"], ["e"]]

    labels = [0, 1, 1, 0, 1, 0]
    grouped_tokens = group_tokens_by_labels(tokens, labels, join=True)
    assert grouped_tokens == ["b", "c", "e"]

    labels = [0, 1, 2, 0, 1, 0]
    grouped_tokens = group_tokens_by_labels(tokens, labels, join=True)
    assert grouped_tokens == ["bc", "e"]


def test_join_punctuation():
    token_list = [
        "This",
        "is",
        "a",
        "list",
        "of",
        "tokens",
        "with",
        "2",
        ".",
        "5",
        ",",
        "and",
        "3",
        "numbers",
        "intact",
        "semi",
        "-",
        "colon",
        "separated",
        "words",
        "with",
        "decimal",
        "numbers",
        "split",
        "at",
        "dots",
        ".",
        "This",
        "is",
        "a",
        "comma",
        ",",
        "and",
        "a",
        "dot",
        "(",
        "test",
        ")",
        ".",
    ]
    sentence = join_punctuation(token_list)
    print(sentence)
    assert (
        sentence
        == "This is a list of tokens with 2.5, and 3 numbers intact semi-colon separated words with decimal numbers split at dots. This is a comma, and a dot (test)."
    )
