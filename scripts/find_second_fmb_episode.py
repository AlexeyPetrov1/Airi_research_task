"""Inspect a few nearby raw FMB episode files without downloading the dataset."""

from huggingface_hub import HfApi


REPO = "charlesxu0124/functional-manipulation-benchmark"
PREFIX = "single_object_manipulation_dataset/1_M_L_3_vertical_n_"


def main() -> None:
    api = HfApi()
    info = api.repo_info(REPO, repo_type="dataset")
    print("revision", info.sha)
    for index in range(1, 11):
        path = f"{PREFIX}{index}.npy"
        print(path, api.file_exists(REPO, path, repo_type="dataset"), flush=True)


if __name__ == "__main__":
    main()
