from pathlib import Path

import hydra
import kagglehub
import pandas as pd
from omegaconf import DictConfig
from PIL import Image
from tqdm import tqdm


@hydra.main(version_base="1.3", config_path="configs", config_name="config.yaml")
def main(cfg: DictConfig):
    kaggle_dataset_path = f"awsaf49/vinbigdata-{cfg.dim}-image-dataset"
    datapath = kagglehub.dataset_download(kaggle_dataset_path)
    datapath = Path(datapath) / "vinbigdata"
    print(f"Path to dataset files: {datapath}")

    train_df_path = datapath / "train.csv"
    train_data_path = datapath / "train"
    test_data_path = datapath / "test"

    train_df = pd.read_csv(train_df_path)

    train_df["x_min"] = train_df["x_min"] / train_df["width"]
    train_df["y_min"] = train_df["y_min"] / train_df["height"]
    train_df["x_max"] = train_df["x_max"] / train_df["width"]
    train_df["y_max"] = train_df["y_max"] / train_df["height"]

    train_df["x_mid"] = (train_df["x_max"] + train_df["x_min"]) / 2
    train_df["y_mid"] = (train_df["y_max"] + train_df["y_min"]) / 2

    train_df["w"] = train_df["x_max"] - train_df["x_min"]
    train_df["h"] = train_df["y_max"] - train_df["y_min"]
    train_df["area"] = train_df["w"] * train_df["h"]

    def agg_func(group):
        return pd.Series(
            {
                "bounding_boxes": group[
                    ["x_min", "y_min", "x_max", "y_max", "x_mid", "y_mid"]
                ].values.tolist(),
                "class_ids": group["class_id"].tolist(),
            }
        )

    print("Aggregating train data")
    train_df = (
        train_df.groupby("image_id", group_keys=False)
        .apply(agg_func, include_groups=False)
        .reset_index()
    )

    output_dir = Path(__file__).absolute().resolve().parent / f"dataset_{cfg.dim}"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_train_dir = output_dir / "train"
    (output_train_dir / "labels").mkdir(parents=True, exist_ok=True)
    (output_train_dir / "images").mkdir(parents=True, exist_ok=True)

    for _, row in tqdm(
        train_df.iterrows(),
        desc="Preparing train data",
        total=len(train_df),
    ):
        image_id = row["image_id"]
        class_ids = row["class_ids"]
        bounding_boxes = row["bounding_boxes"]

        label_path = output_train_dir / "labels" / f"{image_id}.txt"
        image_path = output_train_dir / "images" / f"{image_id}.png"

        image = Image.open(train_data_path / f"{image_id}.png")
        image.save(image_path)

        with label_path.open(mode="w") as f:
            for class_id, bbox in zip(class_ids, bounding_boxes):
                line = f"{class_id} " + " ".join(map(str, bbox)) + "\n"
                f.write(line)
    print(f"All train files have been created in the {output_train_dir} directory")

    output_dir_test = output_dir / "test/images"
    output_dir_test.mkdir(parents=True, exist_ok=True)
    image_paths = list(test_data_path.glob("*.png"))
    for image_path in tqdm(image_paths, desc="Preparing test data"):
        image = Image.open(image_path)
        image.save(output_dir_test / image_path.name)
    print(f"All test files have been created in the {output_dir_test} directory")


if __name__ == "__main__":
    main()
