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

# print(torch.__version__, torch.cuda.is_available())

# Check MMSegmentation installation
# print(mmseg.__version__)
config_file = "mmsegmentation/configs/pspnet/pspnet_r50-d8_512x1024_40k_cityscapes.py"
checkpoint_file = "mmsegmentation/checkpoints/pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth"
# build the model from a config file and a checkpoint file
model = init_segmentor(config_file, checkpoint_file, device="cuda:0")
# test a single image
img = "/home/keshav/car.jpg"
result, prob_per_pixel = inference_segmentor(model, img)
result1 = np.array(result).squeeze()
print(result1.shape, result1)
# prob = prob_per_pixel.cpu().numpy().squeeze()
u, c = np.unique(result1, return_counts=True)
needed_class = u[c == c.max()]
result1[result1 != needed_class] = 0
print(np.max(result1), result1.shape)
result1[result1 > 0] = 255
output = np.empty((result1.shape[0], result1.shape[1], 3))
for i in range(result1.shape[0]):
    for j in range(result1.shape[1]):
        if result1[i][j] == 0:
            output[i][j] = np.array([0, 0, 0])
        else:
            output[i][j] = np.array([255, 0, 0])

print(output.shape)
cv2.imshow("result1", output)
cv2.waitKey(0)
cv2.destroyAllWindows()
exit()
# np.set_printoptions(threshold=None)
# with open('GFG.csv', 'w') as f:

#     # using csv.writer method from CSV package
#     write = csv.writer(f)

#     write.writerow(result1)
# np.set_printoptions(threshold=sys.maxsize)
# print("len===", len(result), len(result[0]), len(result[0][0]))
# print(result1)
# with open('list.txt', 'w') as f:
#     for line in result1:
#         f.write(f"{line}\n")
# show the results
# show_result_pyplot(model, img, result, get_palette('cityscapes'))


# def show_result_pyplot(model,
#                        img,
#                        result,
#                        palette=None,
#                        fig_size=(15, 10),
#                        opacity=0.5,
#                        title='',
#                        block=True,
#                        out_file=None):
#     """Visualize the segmentation results on the image.

#     Args:
#         model (nn.Module): The loaded segmentor.
#         img (str or np.ndarray): Image filename or loaded image.
#         result (list): The segmentation result.
#         palette (list[list[int]]] | None): The palette of segmentation
#             map. If None is given, random palette will be generated.
#             Default: None
#         fig_size (tuple): Figure size of the pyplot figure.
#         opacity(float): Opacity of painted segmentation map.
#             Default 0.5.
#             Must be in (0, 1] range.
#         title (str): The title of pyplot figure.
#             Default is ''.
#         block (bool): Whether to block the pyplot figure.
#             Default is True.
#         out_file (str or None): The path to write the image.
#             Default: None.
#     """
#     if hasattr(model, 'module'):
#         model = model.module
#     img = model.show_result(
#         img, result, palette=palette, show=False, opacity=opacity)
#     plt.figure(figsize=fig_size)
#     plt.imshow(mmcv.bgr2rgb(img))
#     plt.title(title)
#     plt.tight_layout()
#     plt.show(block=block)
#     if out_file is not None:
#         mmcv.imwrite(img, out_file)


# def inference_segmentor(model, imgs):
#     """Inference image(s) with the segmentor.

#     Args:
#         model (nn.Module): The loaded segmentor.
#         imgs (str/ndarray or list[str/ndarray]): Either image files or loaded
#             images.

#     Returns:
#         (list[Tensor]): The segmentation result.
#     """
#     cfg = model.cfg
#     device = next(model.parameters()).device  # model device
#     # build the data pipeline
#     test_pipeline = [LoadImage()] + cfg.data.test.pipeline[1:]
#     test_pipeline = Compose(test_pipeline)
#     # prepare data
#     data = []
#     imgs = imgs if isinstance(imgs, list) else [imgs]
#     for img in imgs:
#         img_data = dict(img=img)
#         img_data = test_pipeline(img_data)
#         data.append(img_data)
#     data = collate(data, samples_per_gpu=len(imgs))
#     if next(model.parameters()).is_cuda:
#         # scatter to specified GPU
#         data = scatter(data, [device])[0]
#     else:
#         data['img_metas'] = [i.data[0] for i in data['img_metas']]

#     # forward the model
#     with torch.no_grad():
#         result = model(return_loss=False, rescale=True, **data)
#     return result
