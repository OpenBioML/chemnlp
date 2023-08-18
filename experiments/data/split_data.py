import datasets
import os
import argparse

datasets.disable_caching()
OUT_DIR = "/fsx/proj-chemnlp/llched-raw-tmp"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "data_path", help="The full path to the original dataset.", type=str
    )
    parser.add_argument(
        "val_test_splits_size",
        help="The total test size for validation / test.",
        type=float,
    )
    args = parser.parse_args()
    name = "papers" if "papers" in args.data_path else "abstracts"
    print("loadinggg")
    ds = datasets.load_dataset(
        "json",
        **{"data_files": {"train": args.data_path}, "split": "train"},
        num_proc=os.cpu_count(),
        keep_in_memory=True,
    )

    print("filteringgg")
    ds = ds.filter(
        lambda x: isinstance(x["sentences"], str) and x["pubDate"] is not None,
        num_proc=os.cpu_count(),
    )

    print("sortinggg")
    index_keys = [(i, x) for i, x in enumerate(ds["pubDate"])]
    sorted_rows = sorted(index_keys, key=lambda x: x[1])
    sorted_indicies = [x[0] for x in sorted_rows]
    sorted_ds = ds.select(sorted_indicies)

    print("making splits")
    train_and_future_val_test = sorted_ds.train_test_split(
        test_size=args.val_test_splits_size, shuffle=False
    )
    train_and_random_val_test = train_and_future_val_test["train"].train_test_split(
        test_size=args.val_test_splits_size
    )
    val_and_test_future_splits = train_and_future_val_test["test"].train_test_split(
        test_size=0.50, shuffle=False
    )
    val_and_test_random_splits = train_and_random_val_test["test"].train_test_split(
        test_size=0.50
    )

    print("saving splits")
    train_and_random_val_test["train"].save_to_disk(
        f"{OUT_DIR}/train_{name}_v1", num_proc=os.cpu_count()
    )
    val_and_test_future_splits["train"].save_to_disk(
        f"{OUT_DIR}/val_{name}_v1_future",
        num_proc=os.cpu_count(),
    )
    val_and_test_future_splits["test"].save_to_disk(
        f"{OUT_DIR}/test_{name}_v1_future",
        num_proc=os.cpu_count(),
    )
    val_and_test_random_splits["train"].save_to_disk(
        f"{OUT_DIR}/val_{name}_v1_random",
        num_proc=os.cpu_count(),
    )
    val_and_test_random_splits["test"].save_to_disk(
        f"{OUT_DIR}/test_{name}_v1_random",
        num_proc=os.cpu_count(),
    )

    print("printing row counts per datasets")
    for x in [
        sorted_ds,
        train_and_random_val_test["train"],
        val_and_test_future_splits["train"],
        val_and_test_future_splits["test"],
        val_and_test_random_splits["train"],
        val_and_test_random_splits["test"],
    ]:
        print(x.num_rows)
