
import h5py
import numpy as np
import torch
import os
from torchvision import transforms
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib import cm as CM

import model


# 图像的初始化操作
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def test_image(img_path):
    #加载模型
    net = model.CSRNet()
    #net = net.cuda()
    net.load_state_dict(torch.load("./CSRNet_0032.pt"))

    # 测试单张图片
    img = transform(Image.open(img_path).convert('RGB'))

    # img = 255.0 * F.to_tensor(Image.open(img_paths).convert('RGB'))
    # img[0, :, :] = img[0, :, :] - 92.8207477031
    # img[1, :, :] = img[1, :, :] - 95.2757037428
    # img[2, :, :] = img[2, :, :] - 104.877445883

    output = net(img.unsqueeze(0))

    print("预测人数为：", output.data.numpy().sum())
    #去掉第一维和第二维
    density_map = np.asarray(output.detach().cpu().reshape(output.detach().cpu().shape[2], output.detach().cpu().shape[3]))
    plt.imshow(density_map, cmap=CM.jet)
    plt.show()


def show(img_path):
    #展示原始图片
    plt.imshow(Image.open(img_path))
    plt.show()
    #展示密度图
    gt_path=img_path.replace('.jpg', '.h5')
    if os.path.exists(gt_path) == True: #如果文件存在
        gt_file = h5py.File(img_path.replace('.jpg', '.h5'), 'r')
        groundtruth = np.asarray(gt_file['density'])
        plt.imshow(groundtruth, cmap=CM.jet)
        plt.show()
        print('真实人数为：',np.sum(groundtruth))


if __name__ == '__main__':
    img_path = './classroom2.jpg'
    test_image(img_path)
    show(img_path)






