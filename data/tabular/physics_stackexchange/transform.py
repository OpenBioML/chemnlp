import pandas as pd
from datasets import load_dataset


def remove_repeated_almost_empty_lines(text):
    # in text remove lines that are almost empty and repeated
    # almost empty means that there is only punctuation, or special characters or spaces
    # repeated means that the line is the same as the previous one
    # return the text without the repeated almost empty lines
    # found by manual inspection that this is an issue in some of the raw text
    lines = text.split("\n")
    new_lines = []
    previous_line = ""
    for line in lines:
        if line.strip() == "":
            continue
        if line.strip() == previous_line:
            continue
        new_lines.append(line)
        previous_line = line.strip()
    return "\n".join(new_lines)


def get_clean_df():
    dataset = load_dataset("marianna13/physics-stackexchange")
    questions_w_answer = []
    df = dataset["train"].to_pandas()
    # we do the following, if there is no answer, we drop this question
    # if there is one answer, we keep it
    # if there are multiple we keep the ones that do not have a score of 0
    # the answers are in an array of arrays, the first element is the answer, the second is the score
    # we then also only keep two columns, the question and the answer, both as string on which we also
    # call the strip function to remove leading and trailing whitespaces
    for _i, row in df.iterrows():
        # skip question with markdown image tag in it
        if "![" in row["question_text"]:
            continue
        if "](" in row["question_text"]:
            continue
        if "http" in row["question_text"]:
            continue
        if len(row["answers"]) == 0:
            continue
        if len(row["answers"]) == 1:
            # if image tag in answer, skip
            if "![" in row["answers"][0][0]:
                continue
            if "](" in row["answers"][0][0]:
                continue
            # if link in answer, skip
            if "http" in row["answers"][0][0]:
                continue
            questions_w_answer.append(
                [
                    row["question_title"],
                    remove_repeated_almost_empty_lines(row["question_text"].strip()),
                    remove_repeated_almost_empty_lines(row["answers"][0][0].strip()),
                ]
            )
        else:
            for answer in row["answers"]:
                if answer[1] != 0:
                    # if image tag in answer, skip
                    if "![" in answer[0]:
                        continue
                    if "](" in answer[0]:
                        continue
                    # if link in answer, skip
                    if "http" in answer[0]:
                        continue
                    questions_w_answer.append(
                        [
                            row["question_title"],
                            remove_repeated_almost_empty_lines(
                                row["question_text"].strip()
                            ),
                            remove_repeated_almost_empty_lines(answer[0].strip()),
                        ]
                    )
                    break

    # we then create a dataframe from the list of questions and answers
    df_qa = pd.DataFrame(questions_w_answer, columns=["title", "q", "a"])
    print(len(df_qa))
    df_qa.to_csv("data_clean.csv", index=False)


if __name__ == "__main__":
    get_clean_df()
