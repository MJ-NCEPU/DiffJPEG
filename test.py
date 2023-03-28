import argparse
import random
import torch
from DiffJPEG import DiffJPEG
import cv2
import numpy as np


def imread_uint(path, n_channels=3):
    #  input: path
    # output: HxWx3(RGB or GGG), or HxWx1 (G)
    if n_channels == 1:
        img = cv2.imread(path, 0)  # cv2.IMREAD_GRAYSCALE
        img = np.expand_dims(img, axis=2)  # HxWx1
    elif n_channels == 3:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # BGR or G
        print("ndim:", img.ndim)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)  # GGG
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # RGB
    return img


def imsave(img, img_path):
    img = np.squeeze(img)
    if img.ndim == 3:
        img = img[:, :, [2, 1, 0]]
    cv2.imwrite(img_path, img)



parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int)
arg = parser.parse_args()

arg.batch_size = 30

#quality 可为int型或者Tensor list 型
# int型,产生一个压缩因子
# quality = 20

# Tensor list 型,产生batch_size个压缩因子
quality = random.sample(range(40, 95), arg.batch_size)
quality = torch.Tensor(quality)

print("quality_factor:", quality)

img = imread_uint('D:/G/pythonworkspace/DiffJPEG-master-batch/test.png', 3)  # [0-255]hwc
img = torch.Tensor(img).unsqueeze(0).permute(0, 3, 1, 2)  # 1chw
img = img.repeat(arg.batch_size, 1, 1, 1) #bchw
B, C, H, W = img.size()

jpeg = DiffJPEG(batch=B, height=H, width=W, differentiable=True, quality=quality, arg=arg)

output = jpeg(img).permute(0, 2, 3, 1)  # bhwc
out = output.detach().numpy()

print("out", type(out), out.shape, out[0][0][0][0])
print("len(out):", len(out))

for i in range(len(out)):
    imsave(out[i], f'./image/test{i}.png')
