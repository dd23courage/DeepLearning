import tqdm
import os
import numpy as np
import scipy
import scipy.io as io
from scipy import spatial
from scipy.ndimage.filters import gaussian_filter
import glob
from matplotlib import pyplot as plt
import h5py

#高斯核函数
def gaussian_filter_density(gt):
    print(gt.shape)

    density = np.zeros(gt.shape, dtype=np.float32)
    gt_count = np.count_nonzero(gt)
    if gt_count == 0:
        return density
    pts = np.array(list(zip(np.nonzero(gt)[1], np.nonzero(gt)[0])))

    #构造KDTree寻找相邻的人头位置
    tree = scipy.spatial.KDTree(pts.copy(), leafsize=2048)
    distances, locations = tree.query(pts, k=4)

    print('generate density...')
    for i, pt in enumerate(pts):
        pt2d = np.zeros(gt.shape, dtype=np.float32)
        pt2d[pt[1],pt[0]] = 1.
        if gt_count > 1:
            #相邻三个人头的平均距离，其中beta=0.3
            sigma = (distances[i][1]+distances[i][2]+distances[i][3])*0.1
        else:
            sigma = np.average(np.array(gt.shape))/2./2. #case: 1 point
        density += scipy.ndimage.filters.gaussian_filter(pt2d, sigma, mode='constant')
    print('done.')
    return density

#生成密度图
def create_ground_truth_density(path_sets):
    img_paths = []

    for path in path_sets:
        for img_path in glob.glob(os.path.join(path, '*.jpg')):
            img_paths.append(img_path)
    print('图片数量：', len(img_paths))

    for img_path in img_paths:
        print(img_path)
        # 获取每张图片对应的mat标记文件
        mat = io.loadmat(img_path.replace('images', 'ground_truth').replace('IMG_', 'GT_IMG_').replace('.jpg', '.mat'))
        img = plt.imread(img_path)
        # 生成密度图
        gt_density_map = np.zeros((img.shape[0], img.shape[1]))
        gt = mat["image_info"][0, 0][0, 0][0]
        for i in range(0, len(gt)):
            if int(gt[i][1]) < img.shape[0] and int(gt[i][0]) < img.shape[1]:
                gt_density_map[int(gt[i][1]), int(gt[i][0])] = 1
        gt_density_map = gaussian_filter_density(gt_density_map)
        # 保存生成的密度图
        with h5py.File(img_path.replace('images', 'ground_truth').replace('.jpg', '.h5'), 'w') as hf:
            hf['density'] = gt_density_map

        #测试
        print('总数量=',len(gt))
        print('密度图=',gt_density_map.sum())


# 查看原始图片和生成的密度图
def show(img_path):
    from PIL import Image
    from matplotlib import cm as CM

    plt.imshow(Image.open(img_path))
    plt.show()
    gt_file = h5py.File(img_path.replace('.jpg', '.h5').replace('images', 'ground_truth'), 'r')
    groundtruth = np.asarray(gt_file['density'])
    plt.imshow(groundtruth, cmap=CM.jet)
    plt.show()
    print('总人数为：',np.sum(groundtruth))


if __name__ == '__main__':
    # set the root to the Shanghai dataset you download
    root = './shanghai_data/'

    # now generate the ShanghaiA's ground truth
    part_A_train = os.path.join(root, 'part_A_final/train_data', 'images')
    part_A_test = os.path.join(root, 'part_A_final/test_data', 'images')
    part_B_train = os.path.join(root, 'part_B_final/train_data', 'images')
    part_B_test = os.path.join(root, 'part_B_final/test_data', 'images')
    path_sets = [part_B_train, part_B_test]  # 将训练集和测试集放在一起

    create_ground_truth_density(path_sets)

    #create_ground_truth_density(path_sets) #生成密度图
    show(os.path.join(part_A_test, 'IMG_1.jpg'))











