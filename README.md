
# PyTorch YOLO on Turfgrass Divots

This project is based on PyTorch-YOLOv3, branched from the master code by Erik Lindernoren. The code has been adapted and modified for the training and testing of a divot dataset.

[Link to Zenodo Record](https://www.zenodo.org/record/8375419)

[![CI](https://github.com/eriklindernoren/PyTorch-YOLOv3/actions/workflows/main.yml/badge.svg)](https://github.com/eriklindernoren/PyTorch-YOLOv3/actions/workflows/main.yml) [![PyPI pyversions](https://img.shields.io/pypi/pyversions/pytorchyolo.svg)](https://pypi.python.org/pypi/pytorchyolo/) [![PyPI license](https://img.shields.io/pypi/l/pytorchyolo.svg)](LICENSE)

## Setup

### Environment Setup for Divot Code

The project was modified to use a conda environment, detailed in the `conda_setup.yml` file. Follow these steps to set up the environment:

1. **Install Conda**: If you haven't installed conda yet, download and install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/products/distribution).

2. **Navigate to the Project Directory**: Use the terminal or command prompt to navigate to the directory containing the `conda_setup.yml` file.

```bash
conda env create -f conda_setup.yml
conda activate environment_name
```

### Divot Code Installation from Source

```bash
git clone <repo_url>
cd PyTorch-YOLOv3/
```

#### Download Pretrained Weights

```bash
./weights/download_weights.sh
```

#### Download COCO Dataset & Divots Data

[Link to Zenodo Record](https://www.zenodo.org/record/8375419)

```bash
./data/get_coco_dataset.sh
```

## Test

Evaluate the model on the COCO test dataset. To download this dataset and weights, see the steps above.

```bash
python test.py --weights weights/yolov3.weights --model config/yolov3.cfg
```

Run detection on sample images:

```bash
python detect.py --images data/samples/
```

<p align="center"><img src="https://github.com/stevefoy/PyTorch-YOLOv3/raw/master/assets/divot.png" width="360"/></p>

#### Example (COCO)

To train on COCO using a Darknet-53 backend pretrained on ImageNet, run:

```bash
python train.py --data config/coco.data --pretrained_weights weights/darknet53.conv.74
```


## Train on Divot Dataset

#### Custom Model

Run the commands below to create a custom model definition, replacing `<num-classes>` with the number of classes in your dataset:

```bash
./config/create_custom_model.sh <num-classes>  # Will create custom model 'yolov3-custom.cfg'
```

#### Classes

Add class names to `data/custom/classes.names`. This file should have one row per class name.

#### Image Folder

Move the images of your dataset to `data/custom/images/`.

#### Annotation Folder

Move your annotations to `data/custom/labels/`. The dataloader expects the annotation file corresponding to the image `data/custom/images/train.jpg` to be at `data/custom/labels/train.txt`. Each row in the annotation file should define one bounding box, using the syntax `label_idx x_center y_center width height`. The coordinates should be scaled `[0, 1]`, and the `label_idx` should be zero-indexed and correspond to the row number of the class name in `data/custom/classes.names`.

#### Define Train and Validation Sets

In `data/custom/train.txt` and `data/custom/valid.txt`, add paths to images that will be used as train and validation data, respectively.

#### Train

To train on the custom dataset, run:

```bash
python train.py --model config/yolov3_divots_416.cfg --data config/custom.data
```

Add `--pretrained_weights weights/darknet53.conv.74` to train using a backend pretrained on ImageNet.

## Credit

**YOLOv3: An Incremental Improvement**
[[Paper]](https://pjreddie.com/media/files/papers/YOLOv3.pdf) [[Project Webpage]](https://pjreddie.com/darknet/yolo/) [[Authors' Implementation]](https://github.com/pjreddie/darknet)

**YOEO — You Only Encode Once**
[YOEO](https://github.com/bit-bots/YOEO) extends this repo with the ability to train an additional semantic segmentation decoder. The lightweight example model is mainly targeted towards embedded real-time applications.

## Licensing

- **YOLOv3**: The MIT license is very permissive and allows for both open-source and proprietary use without the need to disclose source code.
- **YOLOv5, YOLOv6, YOLOv7, YOLOv8**: The AGPL-3.0 license requires any derived works to be open-sourced under the same terms, which can be restrictive for commercial applications.

## Citation

```bibtex
@inproceedings{IMVIP2024,
    author = {Stephen Foy and Simon McLoughlin},
    title = {Assessment of Synthetic Turfgrass Dataset Generation for Divot Detection},
    booktitle = {Irish Machine Vision and Image Processing Conference Proceedings 2024},
    editor = {TBC},
    year = {2024},
    address = {Limerick, Ireland},
    publisher = {Irish Pattern Recognition and Classification Society},
    isbn = {TBC},
    pages = {TBC},
}
```
