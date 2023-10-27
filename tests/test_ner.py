from chemnlp.data.ner import group_tokens_by_labels


def test_tokens_by_label():
    tokens = ["a", "b", "c", "d", "e", "f"]
    labels = [0, 1, 1, 0, 1, 0]
    grouped_tokens = group_tokens_by_labels(tokens, labels)
    assert grouped_tokens == [["b"], ["c"], ["e"]]

    labels = [0, 1, 2, 0, 1, 0]
    grouped_tokens = group_tokens_by_labels(tokens, labels)
    assert grouped_tokens == [["b", "c"], ["e"]]
