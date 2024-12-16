# Dataset Structure

This dataset is organized into two primary splits: `train` and `test`. Each split contains subfolders for `images` and, in the case of the training set, `labels`. Below is a detailed description of the folder structure and contents:

## Folder Structure

```
dataset/
    train/
        images/
            image_1.png
            image_2.png
            ...
        labels/
            image_1.txt
            image_2.txt
            ...
    test/
        images/
            image_1.png
            image_2.png
            ...
```

### Train Split

- **`images/`**: Contains the training images in `.png` format. Each image corresponds to a label file in the `labels/` directory.
- **`labels/`**: Contains the label files in `.txt` format. Each label file corresponds to an image in the `images/` directory. The label files describe the annotations for the objects in the images.

### Test Split

- **`images/`**: Contains the test images in `.png` format. This split does not include label files as it is typically used for evaluation purposes.

---

## Label Format

Each `.txt` file in the `labels/` directory contains annotations for the corresponding image. Each line in the file represents a single object in the image with the following format:

```
class_id x_min y_min x_max y_max x_mid y_mid
```

### Details:

- **`class_id`**: The class ID of the object. An integer representing the object category.
- **`x_min`**, **`y_min`**: Normalized coordinates of the top-left corner of the bounding box.
- **`x_max`**, **`y_max`**: Normalized coordinates of the bottom-right corner of the bounding box.
- **`x_mid`**, **`y_mid`**: Normalized coordinates of the center of the bounding box.

All coordinates are normalized to the range `[0, 1]` relative to the image dimensions.

### Example Label Files:

1. **No valid bounding box information**:
   ```
   14 nan nan nan nan nan nan
   14 nan nan nan nan nan nan
   14 nan nan nan nan nan nan
   ```

   This indicates that objects of class `14` are present, but no bounding box or position information is provided.

2. **Bounding box information included**:
   ```
   6 0.314 0.4116666666666667 0.472 0.536 0.393 0.47383333333333333
   4 0.24366666666666667 0.165 0.519 0.4226666666666667 0.38133333333333336 0.29383333333333334
   7 0.6176666666666667 0.22866666666666666 0.7876666666666666 0.474 0.7026666666666667 0.35133333333333333
   ```

   - Line 1: Object of class `6` with bounding box spanning from `(0.314, 0.4117)` to `(0.472, 0.536)` and centered at `(0.393, 0.4738)`.
   - Line 2: Object of class `4` with a bounding box and center coordinates similarly defined.
   - Subsequent lines follow the same structure.

## Dataset Generation

To generate the dataset, you need to run the `generate_dataset.py` script located in the `data/` directory. The script allows you to specify the dimensions of the images to be generated. The available options for dimensions are `256`, `512`, and `1024`.

### Command to Generate the Dataset

1. **Default Behavior**:
   If no dimension is specified, the script will default to generating images with dimensions `256x256`:
   ```bash
   python data/generate_dataset.py
   ```
   Dataset will be saved in `data/dataset_256`.

2. **Specifying Dimensions**:
   To specify the dimensions, use the `dim` parameter when running the script. The generated dataset will be saved in a folder named `data/dataset_{dim}`, where `{dim}` corresponds to the specified dimensions (`256`, `512`, or `1024`). For example:
   ```bash
   python data/generate_dataset.py dim=256
   ```
   Replace `256` with `512` or `1024` if you wish to generate images of those dimensions.

   Example for `512x512` images:
   ```bash
   python data/generate_dataset.py dim=512
   ```
   Dataset will be saved in `data/dataset_256`.

3. **Available Dimensions**:
   - `256` (default)
   - `512`
   - `1024`

### Notes on Dataset Generation
- The script will automatically create the necessary folder structure (`train/images`, `train/labels`, and `test/images`) and populate it with the generated dataset.
- Make sure to have the necessary dependencies installed before running the script.

---

## Notes

- **Coordinate System**: All coordinates are normalized to ensure compatibility across images of varying dimensions.
- **Missing Data**: Lines with `nan` indicate missing or unavailable bounding box information for the respective objects.
- **Use Case**: The dataset is suitable for training and evaluating object detection models, where the `train/labels/` provides annotation data for supervised learning, and the `test/` split can be used for model evaluation or testing.
