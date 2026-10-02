"""Print metadata for a few exact FMB raw files without downloading the dataset."""

from huggingface_hub import HfApi


REPO = "charlesxu0124/functional-manipulation-benchmark"
FILES = [f"single_object_manipulation_dataset/1_M_L_3_vertical_n_{i}.npy"
         for i in range(4, 10)]


def main() -> None:
    for item in HfApi().get_paths_info(REPO, FILES, repo_type="dataset"):
        print(item.path, item.size)


if __name__ == "__main__":
    main()
