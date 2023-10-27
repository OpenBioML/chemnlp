def group_tokens_by_labels(tokens, labels):
    grouped_tokens = []
    current_group = []

    for token, label in zip(tokens, labels):
        if label == 1:
            if current_group:
                grouped_tokens.append(current_group)
            current_group = [token]
        elif label == 2 and current_group:
            current_group.append(token)

    if current_group:
        grouped_tokens.append(current_group)

    return grouped_tokens
