import cv2
import torch
import numpy as np
import segmentation_models_pytorch as smp
from torch.nn.functional import softmax
seg_model = smp.Unet(encoder_name="resnet34",
                     encoder_weights="imagenet", in_channels=3, classes=3)


img = cv2.imread('/home/vicky/cat.jpeg')
print("input shape", img.shape)
rem_32_x, rem_32_y = img.shape[0] % 32, img.shape[1] % 32

img = cv2.resize(img, (480, 544))
img = torch.from_numpy(img).float()
seg_mask = seg_model(img.transpose(2, 0).unsqueeze(dim=0))
print("seg_mask output shape ", seg_mask.shape)
# print(img2)
# cv2.imshow('img', img2)
seg_mask = softmax(seg_mask, dim=1)
img2 = seg_mask.detach().squeeze(dim=0).transpose(1, 0).transpose(1, 2).numpy()
classes = [np.array([255, 0, 0]), np.array([0, 0, 255])]
print(img2.shape, img2[0][0])
output = np.empty((img2.shape[0], img2.shape[1], 3))

for x in range(len(img2)):
    for y in range(len(img2[0])):
        if img2[x][y][0] > img2[x][y][1]:
            output[x][y] = classes[0]
        else:
            output[x][y] = classes[1]
print("output shape", output.shape)
cv2.imshow("output", output)
cv2.waitKey(0)
cv2.destroyAllWindows()
