import re


def group_tokens_by_labels(tokens, labels, join=True):
    grouped_tokens = []
    current_group = []

    for token, label in zip(tokens, labels):
        if label == 1:
            if current_group:
                if join:
                    current_group = "".join(current_group)
                    grouped_tokens.append(current_group)
                else:
                    grouped_tokens.append(current_group)
            current_group = [token]
        elif label == 2 and current_group:
            current_group.append(token)

    if current_group:
        if join:
            current_group = "".join(current_group)
            grouped_tokens.append(current_group)
        else:
            grouped_tokens.append(current_group)

    return grouped_tokens


def join_punctuation(token_list):
    # join tokens based on the following rules
    # if current token is a punctuation, join it with the previous token
    # if current token is a number, join it with the previous token if the previous token is a number
    # or the previous token is a punctuation and the token before the previous token is a number
    # if the current token is a word, join it with the previous token if the previous token is -
    # otherwise, add a space before the current token if the previous token is not a parenthesis or bracket
    # if the previous token is a punctuation, add a space in case the punctuation does not separate
    # two numbers

    output = []

    for i, token in enumerate(token_list):
        if i == 0:
            output.append(token)
            continue

        if token in [
            ".",
            ",",
            ";",
            ":",
            "-",
            "(",
            "[",
            "{",
            "}",
            "]",
            ")",
            "!",
            "/",
            "'",
        ]:
            if re.match(r"\d", token):
                if re.match(r"\d", token_list[i - 1]):
                    output[-1] += token
                elif token_list[i - 1] in [
                    ".",
                    ",",
                    ";",
                    ":",
                    "-",
                    "(",
                    "[",
                    "{",
                ] and re.match(r"\d", token_list[i - 2]):
                    output[-2] += token_list[i - 1]
                    output[-2] += token
                    output.pop(-1)
                else:
                    output.append(token)
            elif token in ["(", "[", "{"] and re.match(r"\w", token_list[i - 1]):
                output.append(" " + token)
            else:
                print(token_list[i - 1], token)
                if token_list[i - 1] in [
                    ".",
                    ",",
                    ";",
                    ":",
                    "-",
                    "(",
                    "[",
                    "{",
                    "}",
                    "]",
                    ")",
                ]:
                    output[-1] += token
                # if token prior to the punctuation is a number, add no space
                elif re.match(r"[\d\w]", token_list[i - 1]):
                    output.append(token)

                else:
                    output.append(" " + token)
        else:
            if token_list[i - 1] in ["!", "/", "-"]:
                output[-1] += token
            elif token_list[i - 1] in [":", "-", "(", "[", "{"]:
                output.append(token)
            elif token_list[i - 1] in ["."]:
                if re.match(r"\d", token_list[i - 2]) and re.match(r"\d", token):
                    output.append(token)
                else:
                    output.append(" " + token)
            else:
                output.append(" " + token)

    return "".join(output)
