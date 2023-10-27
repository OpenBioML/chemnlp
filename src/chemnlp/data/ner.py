import re


def group_tokens_by_labels(tokens, labels, join=True):
    grouped_tokens = []
    current_group = []

    for token, label in zip(tokens, labels):
        if label == 1:
            if current_group:
                if join:
                    current_group = punctuation_joiner(current_group)
                    grouped_tokens.append(current_group)
                else:
                    grouped_tokens.append(current_group)
            current_group = [token]
        elif label == 2 and current_group:
            current_group.append(token)

    if current_group:
        if join:
            current_group = punctuation_joiner(current_group)
            grouped_tokens.append(current_group)
        else:
            grouped_tokens.append(current_group)

    return list(set(grouped_tokens))


def punctuation_joiner(tokens):
    # join tokens with spaces
    joined = " ".join(tokens)
    joined = cleaner(joined)
    return joined


def cleaner(string):
    # join tokens with spaces and then remove space that should not be there
    # spaces that should not be there are:
    # 1. space before punctuation
    # 2. space after left parenthesis, including square and curly brackets
    # 3. space before right parenthesis, including square and curly brackets
    # 4. space before hyphen
    # 5. space after hyphen
    # 6. space before colon
    # 7. space between numbers separated by dots
    # 8. space between / and numbers or letters
    # 9. space ' and letters
    # 10. no space in  + /-  (e.g. 1+/-2) or + /- should be +/-
    # 11. no space in front of +
    # 12 no space after opening quote
    # 13 no space before closing quote, e.g. "test " -> "test"

    # remove space before punctuation
    joined = re.sub(r"\s([.,;!?])", r"\1", string)

    # remove space after left parenthesis
    joined = re.sub(r"([(])\s", r"\1", joined)

    # remove space before right parenthesis
    joined = re.sub(r"\s([)\]}])", r"\1", joined)

    # remove space before hyphen
    joined = re.sub(r"\s(-)", r"\1", joined)

    # remove space after hyphen
    joined = re.sub(r"(-)\s", r"\1", joined)

    # remove space before colon
    joined = re.sub(r"\s(:)", r"\1", joined)

    # remove space between numbers separated by dots, e.g 1. 3 -> 1.3 or 1 . 3 -> 1.3 or 1 .3 -> 1.3
    joined = re.sub(r"(\d)\s(\.)\s(\d)", r"\1\2\3", joined)
    joined = re.sub(r"(\d)\s(\.)(\d)", r"\1\2\3", joined)
    joined = re.sub(r"(\d)(\.)\s(\d)", r"\1\2\3", joined)

    # remove space between / and numbers or letters, e.g. / 3 -> /3 or / 3.5 -> /3.5 or A/ B    -> A/B or A /B -> A/B
    joined = re.sub(r"([A-Za-z0-9])\s(/)\s([A-Za-z0-9])", r"\1\2\3", joined)

    # remove space between ' and letters, e.g. A' s -> A's or A 's -> A's or A ' s -> A's
    joined = re.sub(r"([A-Za-z])\s(')\s([A-Za-z])", r"\1\2\3", joined)

    # remove space in  + /-  (e.g. 1+/-2) or + /- should be +/-
    joined = re.sub(r"(\+|-)\s(/-)", r"\1\2", joined)

    # remove space in front of +
    joined = re.sub(r"\s(\+)", r"\1", joined)

    # replace double quotes "" with "
    joined = re.sub(r'""', '"', joined)

    # replace multiple spaces with single space
    joined = re.sub(r"\s+", " ", joined)

    # remove spaces after the opening quote, make sure we have a closing quote
    # i.e. the regex needs to match \"\s+(anything)"
    joined = re.sub(r'"\s+(.+)"', r'"\1"', joined)

    # remove spaces before the closing quote, make sure we have an opening quote
    # i.e. the regex needs to match "(anything)\s+"
    joined = re.sub(r'"(.+)\s+"', r'"\1"', joined)

    return joined
