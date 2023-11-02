from datasets import load_dataset
import pandas as pd

from pylatexenc.latexencode import unicode_to_latex


def uniCode2Latex(text: str) -> str:
    """
    converts unicode text to latex and
    fixes UTF-8 chars for latex in a certain range:
        ₀:$_0$ ... ₉:$_9$

    see https://github.com/phfaist/pylatexenc/issues/72

    Args:
        text(str): the string to fix

    Return:
        str: latex presentation of UTF-8 char
    """
    for code in range(8320, 8330):
        text = text.replace(chr(code), f"$_{code-8320}$")

    text = text.replace("\u0305", "$^-$")
    text = text.replace("\u207A", "$^+$")
    text = text.replace("\u207B", "$^-$")
    text = text.replace("\u2074", "$^4$")
    text = text.replace("\u2070", "$^0$")
    text = text.replace("\u2078", "$^1$")
    text = text.replace("\u2075", "$^2$")
    text = text.replace("\u2076", "$^3$")
    text = text.replace("\u2077", "$^5$")

    return unicode_to_latex(text)


def process():
    dataset = load_dataset("kjappelbaum/chemnlp-ocp")
    df_train = dataset["train"].to_pandas()
    df_val = dataset["valid"].to_pandas()

    df_train["split"] = "train"
    df_val["split"] = "valid"

    df = pd.concat([df_train, df_val])
    df["text"] = df["text"].apply(uniCode2Latex)
    print(len(df))
    df.to_csv("data_clean.csv", index=False)


if __name__ == "__main__":
    process()
