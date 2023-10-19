# PyTorch YOLO on Turfgrass Divots
This project is based on PyTorch-YOLOv3 and branched from master code eriklindernoren.

The code has been adapted and modified for the training and testing of a divot dataset. 

[Link to Zenodo Record](https://www.zenodo.org/record/8375419)


[![CI](https://github.com/eriklindernoren/PyTorch-YOLOv3/actions/workflows/main.yml/badge.svg)](https://github.com/eriklindernoren/PyTorch-YOLOv3/actions/workflows/main.yml) [![PyPI pyversions](https://img.shields.io/pypi/pyversions/pytorchyolo.svg)](https://pypi.python.org/pypi/pytorchyolo/) [![PyPI license](https://img.shields.io/pypi/l/pytorchyolo.svg)](LICENSE)

## Setup

### Environment for Divot code installing from source

The project was changed to use conda environment, shared in the conda_setup.yml

Follow these steps to set up a conda environment using the provided `conda_setup.yml` file:

1. **Install Conda**: If you haven't installed conda yet, download and install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/products/distribution).

2. **Navigate to the Project Directory**: Use the terminal or command prompt to navigate to the directory containing the `conda_setup.yml` file.

```bash
conda env create -f conda_setup.yml
conda activate environment_name


```


### Divot code installing from source


git clone repo
cd PyTorch-YOLOv3/


```



#### Download pretrained weights

```bash
./weights/download_weights.sh
```

#### Download COCO Dataset & Divots Data
[Link to Zenodo Record](https://www.zenodo.org/record/8375419)

```bash
./data/get_coco_dataset.sh
```

## Test
Evaluates the model on COCO test dataset.
To download this dataset as well as weights, see above.

```bash
python test.py --weights weights/yolov3.weights --model config/yolov3.cfg 
```


```bash
python detect.py --images data/samples/
```

<p align="center"><img src="https://github.com/stevefoy/PyTorch-YOLOv3/raw/master/assets/divot.png" width="360"\></p>


#### Example (COCO)
To train on COCO using a Darknet-53 backend pretrained on ImageNet run:

```bash
poetry run yolo-train --data config/coco.data  --pretrained_weights weights/darknet53.conv.74
```

#### Tensorboard
Track training progress in Tensorboard:
* Initialize training
* Run the command below
* Go to http://localhost:6006/

```bash
poetry run tensorboard --logdir='logs' --port=6006
```

Storing the logs on a slow drive possibly leads to a significant training speed decrease.

You can adjust the log directory using `--logdir <path>` when running `tensorboard` and `yolo-train`.

## Train on Divot Dataset

#### Custom model
Run the commands below to create a custom model definition, replacing `<num-classes>` with the number of classes in your dataset.

```bash
./config/create_custom_model.sh <num-classes>  # Will create custom model 'yolov3-custom.cfg'
```

#### Classes
Add class names to `data/custom/classes.names`. This file should have one row per class name.

#### Image Folder
Move the images of your dataset to `data/custom/images/`.

#### Annotation Folder
Move your annotations to `data/custom/labels/`. The dataloader expects that the annotation file corresponding to the image `data/custom/images/train.jpg` has the path `data/custom/labels/train.txt`. Each row in the annotation file should define one bounding box, using the syntax `label_idx x_center y_center width height`. The coordinates should be scaled `[0, 1]`, and the `label_idx` should be zero-indexed and correspond to the row number of the class name in `data/custom/classes.names`.

#### Define Train and Validation Sets
In `data/custom/train.txt` and `data/custom/valid.txt`, add paths to images that will be used as train and validation data respectively.

#### Train
To train on the custom dataset run:

```bash
poetry run yolo-train --model config/yolov3-custom.cfg --data config/custom.data
```

Add `--pretrained_weights weights/darknet53.conv.74` to train using a backend pretrained on ImageNet.


## API

You are able to import the modules of this repo in your own project if you install the pip package `pytorchyolo`.

An example prediction call from a simple OpenCV python script would look like this:

```python
import cv2
from pytorchyolo import detect, models

# Load the YOLO model
model = models.load_model(
  "<PATH_TO_YOUR_CONFIG_FOLDER>/yolov3.cfg",
  "<PATH_TO_YOUR_WEIGHTS_FOLDER>/yolov3.weights")

# Load the image as a numpy array
img = cv2.imread("<PATH_TO_YOUR_IMAGE>")

# Convert OpenCV bgr to rgb
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Runs the YOLO model on the image
boxes = detect.detect_image(model, img)

print(boxes)
# Output will be a numpy array in the following format:
# [[x1, y1, x2, y2, confidence, class]]
```

For more advanced usage look at the method's doc strings.

## Credit

### YOLOv3: An Incremental Improvement
_Joseph Redmon, Ali Farhadi_ <br>

**Abstract** <br>
We present some updates to YOLO! We made a bunch
of little design changes to make it better. We also trained
this new network that’s pretty swell. It’s a little bigger than
last time but more accurate. It’s still fast though, don’t
worry. At 320 × 320 YOLOv3 runs in 22 ms at 28.2 mAP,
as accurate as SSD but three times faster. When we look
at the old .5 IOU mAP detection metric YOLOv3 is quite
good. It achieves 57.9 AP50 in 51 ms on a Titan X, compared
to 57.5 AP50 in 198 ms by RetinaNet, similar performance
but 3.8× faster. As always, all the code is online at
https://pjreddie.com/yolo/.

[[Paper]](https://pjreddie.com/media/files/papers/YOLOv3.pdf) [[Project Webpage]](https://pjreddie.com/darknet/yolo/) [[Authors' Implementation]](https://github.com/pjreddie/darknet)

```
@article{yolov3,
  title={YOLOv3: An Incremental Improvement},
  author={Redmon, Joseph and Farhadi, Ali},
  journal = {arXiv},
  year={2018}
}
```

## Other

### YOEO — You Only Encode Once

[YOEO](https://github.com/bit-bots/YOEO) extends this repo with the ability to train an additional semantic segmentation decoder. The lightweight example model is mainly targeted towards embedded real-time applications.


### Turfgrass Divot Detection
## Cite
```
 @inproceedings{TBC_2023,
   author = {Stephen Foy and Simon Mc Loughlin},
   title = {Deep Learning for Turfgrass Divot Detection},
   booktitle = { TBC},
   year = {2023}
  }
