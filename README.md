# DiffJPEG
# The image JPEG with pytorch,Support input of Batch.
# The specific information and code in test.py
'''python
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int)
arg = parser.parse_args()
arg.batch_size = 30
#quality 可为int型或者Tensor list 型
#int型,产生一个压缩因子
#quality = 20
#Tensor list 型,产生batch_size个压缩因子
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
'''
