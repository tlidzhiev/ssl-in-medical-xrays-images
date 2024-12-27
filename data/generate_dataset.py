from pathlib import Path

import hydra
import kagglehub
import pandas as pd
from omegaconf import DictConfig
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def prepare_train_val_data(cfg: DictConfig, train_df):
    def agg_func(group):
        return pd.Series(
            {
                "bounding_boxes": group[
                    ["x_min", "y_min", "x_max", "y_max", "x_mid", "y_mid"]
                ].values.tolist(),
                "class_ids": group["class_id"].tolist(),
            }
        )

    train_df["x_min"] = train_df["x_min"] / train_df["width"]
    train_df["y_min"] = train_df["y_min"] / train_df["height"]
    train_df["x_max"] = train_df["x_max"] / train_df["width"]
    train_df["y_max"] = train_df["y_max"] / train_df["height"]

    train_df["x_mid"] = (train_df["x_max"] + train_df["x_min"]) / 2
    train_df["y_mid"] = (train_df["y_max"] + train_df["y_min"]) / 2
    train_df["w"] = train_df["x_max"] - train_df["x_min"]
    train_df["h"] = train_df["y_max"] - train_df["y_min"]
    train_df["area"] = train_df["w"] * train_df["h"]

    print("Aggregating train data")
    train_df = (
        train_df.groupby("image_id", group_keys=False)
        .apply(agg_func, include_groups=False)
        .reset_index()
    )

    no_findings_mask = train_df["class_ids"].apply(lambda x: all(i == 14 for i in x))
    train_no_findings, val_no_findings = train_test_split(
        train_df[no_findings_mask], test_size=cfg.val_size, random_state=42
    )
    train_non_no_findigs, val_non_no_findings = train_test_split(
        train_df[~no_findings_mask], test_size=cfg.val_size, random_state=42
    )

    train_part = pd.concat([train_no_findings, train_non_no_findigs])
    val_part = pd.concat([val_no_findings, val_non_no_findings])
    return train_part, val_part


def save_train_val_data(
    savepath: Path, dataset: pd.DataFrame, datapath: Path, part: str
):
    (savepath / "labels").mkdir(parents=True, exist_ok=True)
    if dataset is not None:
        (savepath / "images").mkdir(parents=True, exist_ok=True)

    for _, row in tqdm(
        dataset.iterrows(),
        desc=f"Saving {part} data",
        total=len(dataset),
    ):
        image_id = row["image_id"]
        class_ids = row["class_ids"]
        bounding_boxes = row["bounding_boxes"]

        label_path = savepath / "labels" / f"{image_id}.txt"
        image_path = savepath / "images" / f"{image_id}.png"

        image = Image.open(datapath / f"{image_id}.png")
        image.save(image_path)

        with label_path.open(mode="w") as f:
            for class_id, bbox in zip(class_ids, bounding_boxes):
                line = f"{class_id} " + " ".join(map(str, bbox)) + "\n"
                f.write(line)
    print(f"All {part} files have been created in the {savepath} directory")


def save_test_data(savepath: Path, datapath: Path, part: str = "test"):
    (savepath / "images").mkdir(parents=True, exist_ok=True)
    image_paths = list(datapath.glob("*.png"))

    for image_path in tqdm(image_paths, desc=f"Saving {part} data"):
        image = Image.open(image_path)
        image.save(savepath / "images" / image_path.name)
    print(f"All test files have been created in the {savepath} directory")


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

    train_part, val_part = prepare_train_val_data(cfg, train_df)

    dataset_dir = Path(__file__).absolute().resolve().parent / f"dataset_{cfg.dim}"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    save_train_val_data(
        savepath=(dataset_dir / "train"),
        dataset=train_part,
        datapath=train_data_path,
        part="train",
    )
    save_train_val_data(
        savepath=(dataset_dir / "val"),
        dataset=val_part,
        datapath=train_data_path,
        part="val",
    )
    save_test_data(savepath=(dataset_dir / "test"), datapath=test_data_path)


if __name__ == "__main__":
    main()
