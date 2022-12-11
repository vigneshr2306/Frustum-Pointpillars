import cv2
import sys
import csv
import tensorflow
import numpy as np
from mmseg.core.evaluation import get_palette
from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot

# import mmseg
import torch
import torchvision
import pickle


config_file = "mmsegmentation/configs/pspnet/pspnet_r50-d8_512x1024_40k_cityscapes.py"
checkpoint_file = "mmsegmentation/checkpoints/pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth"

model = init_segmentor(config_file, checkpoint_file, device="cuda:0")


def bbox_extract(image_path, bbox):
    print(bbox)
    ymax, ymin, xmax, xmin = bbox
    image = cv2.imread(image_path)
    image = image[xmin:xmax, ymin:ymax]
    return image


def segmentation_frustum(image_path, bbox, xy, show=False):
    img = bbox_extract(image_path, bbox)
    segmentation_output, prob_per_pixel = inference_segmentor(model, img)
    segmentation_output = np.array(segmentation_output).squeeze()

    prob_per_pixel = (
        prob_per_pixel.cpu().squeeze().transpose(0, 1).transpose(1, 2).numpy()
    )
    print(segmentation_output.shape, prob_per_pixel.shape)
    unique_class, count = np.unique(segmentation_output, return_counts=True)
    needed_class = unique_class[count == count.max()]
    segmentation_output[segmentation_output != needed_class] = 0
    segmentation_output[segmentation_output > 0] = 255
    output = np.empty((segmentation_output.shape[0], segmentation_output.shape[1], 3))
    prob_output = np.empty(
        (segmentation_output.shape[0], segmentation_output.shape[1], 1)
    )
    for i in range(segmentation_output.shape[0]):
        for j in range(segmentation_output.shape[1]):
            if segmentation_output[i][j] == 0:
                output[i][j] = np.array([0, 0, 0])
                prob_output[i][j] = 0
            else:
                output[i][j] = np.array([255, 0, 0])
                prob_output[i][j] = prob_per_pixel[i][j][needed_class]
    # print(prob_output)
    # print(prob_output.shape, prob_output)
    # l = np.array([prob_output[[120, 130, 140, 150], [120, 120, 120, 120]]])
    print(xy[:, 0], xy[:, 1], xy.shape)
    l = np.array([prob_output[xy[:, 0], xy[:, 1]]]).squeeze()
    print(l)

    if show:
        cv2.imshow("segmentation_output", output)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    print("here")
    img_path = "/home/keshav/car.jpg"
    bbox = (480, 0, 293, 0)
    xy = np.array([[128, 120], [130, 120], [140, 124], [150, 126]])
    segmentation_frustum(img_path, bbox, xy, show=False)
