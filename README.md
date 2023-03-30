# DiffJPEG 支持Batch批量处理和单个处理，运行请在test.py,并修改读取图片路径。
# The image JPEG with pytorch,Support input of Batch。
# The specific information and code in *test.py*。
# The input image should be a square and its H and W should be a multiple of 8。
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
    W_ = W
    H_ = H

    h_pad = 0
    w_pad = 0
    if H % 16 != 0:
        h_pad = 16 - H % 16
        H = H + h_pad
    if W % 16 != 0:
        w_pad = 16 - W % 16
        W = W + w_pad

    img = F.pad(img, (0, w_pad, 0, h_pad), mode='constant', value=0)
    
    jpeg = DiffJPEG(batch=B, height=H, width=W, differentiable=True, quality=quality, arg=arg)
    output = jpeg(img)[:, :, 0:H_, 0:W_].permute(0, 2, 3, 1)  # bhwc
    
    out = output.detach().numpy()
    print("out", type(out), out.shape, out[0][0][0][0])
    print("len(out):", len(out))
    for i in range(len(out)):
        imsave(out[i], f'./image/test{i}.png')
![样例](https://github.com/MJ-NCEPU/DiffJPEG/blob/main/sample1.png)
