
import os
import glob
import numpy as np
import h5py
import cv2

import torch
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
from PIL import Image

import model


# 图像的初始化操作
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 首先继承上面的dataset类。然后在__init__()方法中得到图像的路径，将图像路径组成一个数组，这样在__getitim__()中就可以直接读取：
class MyDataset(Dataset):  # 创建自己的类：MyDataset,这个类是继承的torch.utils.data.Dataset
    def __init__(self, img_paths, transform=None, target_transform=None):  # 初始化一些需要传入的参数
        super(MyDataset, self).__init__()  # 对继承自父类的属性进行初始化
        self.img_paths = img_paths
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):  # 这个方法是必须要有的，用于按照索引读取每个元素的具体内容
        img_path = self.img_paths[index] #根据索引获取图片路径
        img = Image.open(img_path).convert('RGB')

        gt_path = img_path.replace('.jpg', '.h5').replace('images', 'ground_truth')
        gt_file = h5py.File(gt_path,'r')
        target = np.asarray(gt_file['density'])
        target = cv2.resize(target, (target.shape[1] // 8, target.shape[0] // 8), interpolation=cv2.INTER_CUBIC) * 64

        if self.transform is not None:
            img = self.transform(img)  # 数据标签转换为Tensor

        return img, target  # return回哪些内容，那么我们在训练时循环读取每个batch时，就能获得哪些内容

    def __len__(self):  # 这个函数也必须要写，它返回的是数据集的长度，也就是多少张图片，要和loader的长度作区分
        return len(self.img_paths)


def load_local_dataset(path_sets, batch_size=8):
    img_paths = []
    path_sets=[path_sets]
    for path in path_sets:
        for img_path in glob.glob(os.path.join(path, '*.jpg')):
            img_paths.append(img_path)
    # 加载数据集
    datasets =MyDataset(img_paths, transform=transform)
    # 调用DataLoader来创建数据集的迭代器
    dataset_iter = DataLoader(dataset=datasets, batch_size=batch_size, shuffle=True)
    return dataset_iter,len(img_paths)


def train(train_iter, net, criterion, optimizer):
    #开始训练
    net.train()  # 启用 BatchNormalization 和 Dropout

    for (img, target) in train_iter:
        img = img.cuda()
        target = torch.unsqueeze(target, 1).cuda()
        output = net(img) #调用模型进行训练

        #损失函数
        loss = criterion(output, target)
        #反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print()
        # print('pre:',output.sum())
        # print('gt:',target.sum())


def validate(test_iter, net, dataset_num):
    net.eval() #开始测试
    mae = 0
    mse = 0

    for (img, target) in test_iter:
        img = img.cuda()
        target = torch.unsqueeze(target, 1).cuda()
        output = net(img)

        mae += abs(output.data.sum() - target.data.sum())
        mse += pow((output.data.sum() - target.data.sum()),2)
    mae = mae / dataset_num
    mse = pow((mse / dataset_num),0.5)

    print('MAE {mae:.3f} MSE {mse:.3f} '.format(mae = mae,mse = mse))


def main():
    # defining the location of dataset
    root = '../dataset/ShanghaiTech_Crowd_Counting_Dataset/'
    part_A_train = os.path.join(root, 'part_A_final/train_data', 'images')
    part_A_test = os.path.join(root, 'part_A_final/test_data', 'images')
    part_B_train = os.path.join(root, 'part_B_final/train_data', 'images')
    part_B_test = os.path.join(root, 'part_B_final/test_data', 'images')

    # 参数设置
    batch_size = 8
    num_epochs = 100
    lr = 1e-7
    momentum = 0.95
    decay = 5 * 1e-4

    #加载数据集
    train_iter,train_num = load_local_dataset(part_B_train,batch_size)
    print('train_num=',train_num)
    test_iter,test_num = load_local_dataset(part_B_test, batch_size)
    print('test_num=',test_num)

    #使用所有GPU进行训练
    net = torch.nn.DataParallel(model.CSRNet()).cuda()

    #定义损失函数和优化器
    loss = torch.nn.MSELoss(reduction='sum').cuda()
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=decay)
    # optimizer_1 = torch.optim.Adam()
    # optimizer_2 = torch.optim.RMSprop()

    for i in range(num_epochs):
        print('第%d次训练' % (i+1) )
        train(train_iter, net, loss, optimizer)
        torch.save(net.module.state_dict(), f'./checkpoint/CSRNet_{str(i+1).zfill(4)}.pt')#保存模型
        validate(test_iter, net,test_num)


if __name__ == '__main__':
    main()


