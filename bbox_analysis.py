#! /usr/bin/env python3

from __future__ import division

import os
import argparse
import tqdm

import torch
from torch.utils.data import DataLoader
import torch.optim as optim

from pytorchyolo.models import load_model
from pytorchyolo.utils.logger import Logger
from pytorchyolo.utils.utils import to_cpu, load_classes, print_environment_info, provide_determinism, worker_seed_set
from pytorchyolo.utils.datasets import ListDataset
from pytorchyolo.utils.augmentations import AUGMENTATION_TRANSFORMS, AUGMENTATION_NONE
#from pytorchyolo.utils.transforms import DEFAULT_TRANSFORMS
from pytorchyolo.utils.parse_config import parse_data_config
from pytorchyolo.utils.loss import compute_loss
from pytorchyolo.test import _evaluate, _create_validation_data_loader
import seaborn as sns; sns.set()  # for plot styling
import numpy as np
import matplotlib.pyplot as plt
    from sklearn.cluster import KMeans

from terminaltables import AsciiTable

from torchsummary import summary


def _create_data_loader(img_path, batch_size, img_size, n_cpu, multiscale_training=False):
    """Creates a DataLoader for training.

    :param img_path: Path to file containing all paths to training images.
    :type img_path: str
    :param batch_size: Size of each image batch
    :type batch_size: int
    :param img_size: Size of each image dimension for yolo
    :type img_size: int
    :param n_cpu: Number of cpu threads to use during batch generation
    :type n_cpu: int
    :param multiscale_training: Scale images to different sizes randomly
    :type multiscale_training: bool
    :return: Returns DataLoader
    :rtype: DataLoader
    """
    dataset = ListDataset(
        img_path,
        img_size=img_size,
        multiscale=multiscale_training,
        transform=AUGMENTATION_NONE)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=n_cpu,
        pin_memory=True,
        collate_fn=dataset.collate_fn,
        worker_init_fn=worker_seed_set)
    
    return dataloader


def run():
    print_environment_info()
    parser = argparse.ArgumentParser(description="Trains the YOLO model.")
    parser.add_argument("-m", "--model", type=str, default="config/yolov3.cfg", help="Path to model definition file (.cfg)")
    parser.add_argument("-d", "--data", type=str, default="config/coco.data", help="Path to data config file (.data)")
    parser.add_argument("-e", "--epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("-v", "--verbose", action='store_true', help="Makes the training more verbose")
    parser.add_argument("--n_cpu", type=int, default=8, help="Number of cpu threads to use during batch generation")
    parser.add_argument("--pretrained_weights", type=str, help="Path to checkpoint file (.weights or .pth). Starts training from checkpoint model")
    parser.add_argument("--checkpoint_interval", type=int, default=1, help="Interval of epochs between saving model weights")
    parser.add_argument("--evaluation_interval", type=int, default=1, help="Interval of epochs between evaluations on validation set")
    parser.add_argument("--multiscale_training", action="store_true", help="Allow multi-scale training")
    parser.add_argument("--iou_thres", type=float, default=0.5, help="Evaluation: IOU threshold required to qualify as detected")
    parser.add_argument("--conf_thres", type=float, default=0.1, help="Evaluation: Object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.5, help="Evaluation: IOU threshold for non-maximum suppression")
    parser.add_argument("--logdir", type=str, default="logs", help="Directory for training log files (e.g. for TensorBoard)")
    parser.add_argument("--seed", type=int, default=-1, help="Makes results reproducable. Set -1 to disable.")
    args = parser.parse_args()
    print(f"Command line arguments: {args}")

    if args.seed != -1:
        provide_determinism(args.seed)

    logger = Logger(args.logdir)  # Tensorboard logger

    # Create output directories if missing
    os.makedirs("output", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    # Get data configuration
    data_config = parse_data_config(args.data)
    train_path = data_config["train"]
    valid_path = data_config["valid"]
    class_names = load_classes(data_config["names"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ############
    # Create model
    # ############

    model = load_model(args.model, args.pretrained_weights)

    # Print model
    if args.verbose:
        summary(model, input_size=(3, model.hyperparams['height'], model.hyperparams['height']))

    mini_batch_size = 1

    # #################
    # Create Dataloader
    # #################

    # Load training dataloader
    dataloader = _create_data_loader(
        train_path,
        mini_batch_size,
        model.hyperparams['height'],
        args.n_cpu,
        args.multiscale_training)

    # Load validation dataloader
    validation_dataloader = _create_validation_data_loader(
        valid_path,
        mini_batch_size,
        model.hyperparams['height'],
        args.n_cpu)

    # ################
    # Create optimizer
    # ################

    params = [p for p in model.parameters() if p.requires_grad]

    if (model.hyperparams['optimizer'] in [None, "adam"]):
        optimizer = optim.Adam(
            params,
            lr=model.hyperparams['learning_rate'],
            weight_decay=model.hyperparams['decay'],
        )
    elif (model.hyperparams['optimizer'] == "sgd"):
        optimizer = optim.SGD(
            params,
            lr=model.hyperparams['learning_rate'],
            weight_decay=model.hyperparams['decay'],
            momentum=model.hyperparams['momentum'])
    else:
        print("Unknown optimizer. Please choose between (adam, sgd).")

    # skip epoch zero, because then the calculations for when to evaluate/checkpoint makes more intuitive sense
    # e.g. when you stop after 30 epochs and evaluate every 10 epochs then the evaluations happen after: 10,20,30
    # instead of: 0, 10, 20

    w_normalized,h_normalized = [] , []

    for epoch in range(1, args.epochs+1):

        print("\n---- Training Model ----")

        model.train()  # Set model to training mode
        
        
        for batch_i, (fileName, imgs, targets) in enumerate(tqdm.tqdm(dataloader, desc=f"Training Epoch {epoch}")):



            batches_done = len(dataloader) * epoch + batch_i

            

            

            # int(obj_ids[index]), yolo_center_x, yolo_center_y, yolo_w, yolo_h 

            img_w, image_h = 416, 416
            # print(type(targets), " ", targets.shape )
            for row in targets:
                    [_,_,x,y,w,h] = row.tolist()
                    # print(x," ", y, " ", w," ", h)
                    w_normalized.append(w)
                    h_normalized.append(h)

            
    w_normalized=np.asarray(w_normalized)
    h_normalized=np.asarray(h_normalized)


    x=[w_normalized, h_normalized]
    x=np.asarray(x)
    x=x.transpose()

    ##########################################   
    #         K- Means
    ##########################################


    kmeans3 = KMeans(n_clusters=9)
    kmeans3.fit(x)
    y_kmeans3 = kmeans3.predict(x)

    ##########################################
    centers3 = kmeans3.cluster_centers_

    yolo_anchor_average=[]
    for ind in range (9):
        yolo_anchor_average.append(np.mean(x[y_kmeans3==ind],axis=0))

    yolo_anchor_average=np.array(yolo_anchor_average)

    YoloScale = 416
    plt.scatter(x[:, 0], x[:, 1], c=y_kmeans3, s=2, cmap='viridis')
    plt.scatter(yolo_anchor_average[:, 0], yolo_anchor_average[:, 1], c='red', s=50);
    yoloV3anchors = yolo_anchor_average
    yoloV3anchors[:, 0] =yolo_anchor_average[:, 0]  *YoloScale
    yoloV3anchors[:, 1] =yolo_anchor_average[:, 1]  *YoloScale
    yoloV3anchors = np.rint(yoloV3anchors)
    fig, ax = plt.subplots()
    for ind in range(9):
        rectangle= plt.Rectangle((208-yoloV3anchors[ind,0]/2,208-yoloV3anchors[ind,1]/2), yoloV3anchors[ind,0],yoloV3anchors[ind,1] , fc='b',edgecolor='b',fill = None)
        ax.add_patch(rectangle)
    ax.set_aspect(1.0)
    plt.axis([0,YoloScale,0,YoloScale])
    plt.show()
    yoloV3anchors.sort(axis=0)
    print("Your custom anchor boxes are {}".format(yoloV3anchors))

            #imgs = imgs.to(device, non_blocking=True)
            #targets = targets.to(device)


            


if __name__ == "__main__":
    run()
