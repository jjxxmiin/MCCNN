import tensorflow as tf
import cv2 as cv
from scipy import misc
from scipy.io import loadmat
import matplotlib.pyplot as plt
import os, random
import numpy as np
from tqdm import tqdm
import request
import math
import zipfile
import glob

# https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/
# https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/

# 배치 만들기
# next batch를 이용해 배치만큼 뽑아낸다
class reader:
    def __init__(self, dir_name, batch_size=None,resize=None):
        self.dir_name = dir_name
        self.batch_size = batch_size
        self.resize = resize
        # file list
        self.file_list = os.listdir(self.dir_name)
        # total batch num
        self.leng = len(self.file_list)
        self.total_batch = self.leng // self.batch_size
        # index
        self.index = 0
        # shuffle
        np.random.shuffle(self.file_list)

    def getList(self):
        return self.file_list

    def getTotalNum(self):
        return len(self.file_list)

    def next_batch(self):
        if self.index == self.total_batch:
            np.random.shuffle(self.file_list)
            self.index = 0

        # image random choice
        batch = []

        file_list_batch = self.file_list[self.index * self.batch_size:(self.index + 1) * self.batch_size]
        self.index += 1

        # 6331번
        for file_name in file_list_batch:
            dir_n = self.dir_name + file_name
            img = misc.imread(dir_n)
            res = misc.imresize(img, self.resize)
            batch.append(res)

        return np.array(batch).astype(np.float32)

# dataset download
class dataset:
    def __init__(self,url,filename):
        self.url = url
        self.filename = filename
        self.current = '//'.join(os.getcwd().split('\\'))+'//'+filename

    def download(self):
        if os.path.exists(self.current):
            print('already existing file')
            return

        wrote = 0
        chunkSize = 1024
        r = request.get(self.url, stream=True)
        total_size = int(r.headers['Content-Length'])
        with open(self.filename, 'wb') as f:
            for data in tqdm(r.iter_content(chunkSize), total=math.ceil(total_size // chunkSize), unit='KB',
                             unit_scale=True):
                wrote = wrote + len(data)
                f.write(data)
        print('download success')
        return

    def upzip(self,savepath='.'):
        if os.path.isdir(self.current[:-4]):
            print('already existing folder')
            return

        with zipfile.ZipFile(self.filename,'r') as zf:
            zf.extractall(path=savepath)
            zf.close()
        print('unzip success')

# test 이미지 불러오기
def load_test_data(image_path, size=256):
    img = misc.imread(image_path, mode='RGB')
    img = misc.imresize(img, [size, size])
    img = np.expand_dims(img, axis=0)
    img = input_normalization(img)

    return img

# 이미지 저장하기
def save_images(images, size, image_path):
    return imsave(input_denormalization(images), size, image_path)

# input_normalization
def input_normalization(img):
    return np.array(img) * (2.0 / 255.0) - 1

def input_denormalization(img):
    return (img + 1.) // 2

# 이미지 저장
def imsave(images, size, path):
    return misc.imsave(path, merge(images, size))

# 이미지 합치기
def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[h*j:h*(j+1), w*i:w*(i+1), :] = image

    return img

# 폴더 있는지 확인
def check_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir

# 배치이미지들 저장
def batch_save(X,nh_nw,path):
    nh, nw = nh_nw
    h, w = X.shape[1], X.shape[2]
    img = np.zeros((h * nh, w * nw, 3))

    for n, x in enumerate(X):
        j = int(n / nw)
        i = int(n % nw)
        img[j * h:j * h + h, i * w:i * w + w, :] = x

    misc.imsave(path,img)

# 붙어있는 이미지 나누기 : pix2pix
def split_image(img):
    tmp = np.split(img,2,axis=2)
    img_A = tmp[0]
    img_B = tmp[1]

    return img_A,img_B


# 모델 불러오기
def load(checkpoint_dir,sess,saver):
    import re
    print(" [*] Reading checkpoints...")

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
        counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
        print(" [*] Success to read {}".format(ckpt_name))
        return True, counter
    else:
        print(" [*] Failed to find a checkpoint")
        return False, 0

# 모델 저장
def save(checkpoint_dir, name, step, sess, saver):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    saver.save(sess, os.path.join(checkpoint_dir, name + '.ckpt'), global_step=step)


def get_file_id(filepath):
    return os.path.splitext(os.path.basename(filepath))[0]

# dataset list 경로 튜플로 저장 : mcnn
def get_data_list(dataset, mode='train'):

    if mode == 'train':
        imagepath = r'G:/ShanghaiTech/part_' + dataset + r'/train_data/images/'
        gtpath = r'G:/ShanghaiTech/part_' + dataset + r'/train_data/ground-truth/'

    elif mode == 'test':
        imagepath = r'G:/ShanghaiTech/part_' + dataset + r'/test_data/images/'
        gtpath = r'G:/ShanghaiTech/part_' + dataset + r'/test_data/ground-truth/'

    image_list = [file for file in glob.glob(os.path.join(imagepath,'*.jpg'))]
    gt_list = []

    for filepath in image_list:
        file_id = get_file_id(filepath)
        gt_file_path = os.path.join(gtpath, 'GT_'+ file_id + '.mat')
        gt_list.append(gt_file_path)

    xy = list(zip(image_list, gt_list))
    # 셔플
    random.shuffle(xy)
    s_image_list, s_gt_list = zip(*xy)

    return s_image_list, s_gt_list, len(s_image_list)


def fspecial(ksize, sigma):
    """
    Generates 2d Gaussian kernel
    :param ksize: an integer, represents the size of Gaussian kernel
    :param sigma: a float, represents standard variance of Gaussian kernel
    :return: 2d Gaussian kernel, the shape is [ksize, ksize]
    """
    # [left, right)
    left = -ksize / 2 + 0.5
    right = ksize / 2 + 0.5
    x, y = np.mgrid[left:right, left:right]
    # generate 2d Gaussian Kernel by normalization
    gaussian_kernel = np.exp(-(np.square(x) + np.square(y)) / (2 * np.power(sigma, 2))) / (2 * np.power(sigma, 2)).sum()
    sum = gaussian_kernel.sum()
    normalized_gaussian_kernel = gaussian_kernel / sum

    return normalized_gaussian_kernel


def get_avg_distance(position, points, k):
    """
    Computes the average distance between a pedestrian and its k nearest neighbors
    :param position: the position of the current point, the shape is [1,1]
    :param points: the set of all points, the shape is [num, 2]
    :param k: a integer, represents the number of mearest neibor we want
    :return: the average distance between a pedestrian and its k nearest neighbors
    """

    # in case that only itself or the k is lesser than or equal to num
    num = len(points)
    if num == 1:
        return 1.0
    elif num <= k:
        k = num - 1

    euclidean_distance = np.zeros((num, 1))
    for i in range(num):
        x = points[i, 1]
        y = points[i, 0]
        # Euclidean distance
        euclidean_distance[i, 0] = math.sqrt(math.pow(position[1] - x, 2) + math.pow(position[0] - y, 2))

    # the all distance between current point and other points
    euclidean_distance[:, 0] = np.sort(euclidean_distance[:, 0])
    avg_distance = euclidean_distance[1:k + 1, 0].sum() / k
    return avg_distance


def get_density_map(scaled_crowd_img_size, scaled_points, knn_phase, k, scaled_min_head_size, scaled_max_head_size):
    """
    Generates the correspoding ground truth density map
    :param scaled_crowd_img_size: the size of ground truth density map
    :param scaled_points: the position set of all points, but were divided into scale already
    :param knn_phase: True or False, determine wheather use geometry-adaptive Gaussian kernel or general one
    :param k: number of k nearest neighbors
    :param scaled_min_head_size: the scaled maximum value of head size for original pedestrian head
                          (in corresponding density map should be divided into scale)
    :param scaled_max_head_size:the scaled minimum value of head size for original pedestrian head
                          (in corresponding density map should be divided into scale)
    :return: density map, the shape is [scaled_img_size[0], scaled_img_size[1]]
    """

    h, w = scaled_crowd_img_size[0], scaled_crowd_img_size[1]

    density_map = np.zeros((h, w))
    # In case that there is no one in the image
    num = len(scaled_points)
    if num == 0:
        return density_map
    for i in range(num):
        # For a specific point in original points label of dataset, it represents as position[oy, ox],
        # so points[i, 1] is x, and points[i, 0] is y Also in case that the negative value
        x = min(h, max(0, abs(int(math.floor(scaled_points[i, 1])))))
        y = min(w, max(0, abs(int(math.floor(scaled_points[i, 0])))))
        # now for a specific point, it represents as position[x, y]

        position = [x, y]

        sigma = 1.5
        beta = 0.3
        ksize = 25
        if knn_phase:
            avg_distance = get_avg_distance(position, scaled_points, k=k)
            avg_distance = max(min(avg_distance, scaled_max_head_size), scaled_min_head_size)
            sigma = beta * avg_distance
            ksize = 1.0 * avg_distance

        # Edge processing
        x1 = x - int(math.floor(ksize / 2))
        y1 = y - int(math.floor(ksize / 2))
        x2 = x + int(math.ceil(ksize / 2))
        y2 = y + int(math.ceil(ksize / 2))

        if x1 < 0 or y1 < 0 or x2 > h or y2 > w:
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(h, x2)
            y2 = min(w, y2)

            tmp = x2 - x1 if (x2 - x1) < (y2 - y1) else y2 - y1
            ksize = min(tmp, ksize)

        ksize = int(math.floor(ksize / 2))
        H = fspecial(ksize, sigma)
        density_map[x1:x1 + ksize, y1:y1 + ksize] = density_map[x1:x1 + ksize, y1:y1 + ksize] + H
    return np.asarray(density_map)


def get_cropped_crowd_image(ori_crowd_img, points, crop_size):
    """
    Crops a sub-crowd image randomly
    :param ori_crowd_img: original crowd image, the shape is [h, w, channel]
    :param points: the original position set of all points
    :param crop_size: the cropped crowd image size we need
    :return: cropped crowd image, cropped points, cropped crowd count
    """
    h, w = ori_crowd_img.shape[0], ori_crowd_img.shape[1]

    # if the original image size < the crooped_image size, reduce the crop size
    if h < crop_size or w < crop_size:
        crop_size = crop_size // 2

    # random to get the crop area
    x1 = random.randint(0, h - crop_size)
    y1 = random.randint(0, w - crop_size)
    x2 = x1 + crop_size
    y2 = y1 + crop_size

    # the crowd image after croppig
    cropped_crowd_img = ori_crowd_img[x1:x2, y1:y2, ...]

    # img_gray_crop = img_gray[x1:x2, y1:y2]
    # img_gray_crop = cv.resize(img_gray_crop, (img_gray_crop.shape[1] // (scale), img_gray_crop.shape[0] // (scale)))

    # Computes the points after cropping
    cropped_points = []
    for i in range(len(points)):
        if x1 <= points[i, 1] <= x2 and y1 <= points[i, 0] <= y2:
            points[i, 0] = points[i, 0] - y1
            points[i, 1] = points[i, 1] - x1
            cropped_points.append(points[i])
    cropped_points = np.asarray(cropped_points)
    cropped_crowd_count = len(cropped_points)
    return cropped_crowd_img, cropped_points, cropped_crowd_count


def get_scaled_crowd_image_and_points(crowd_img, points, scale):
    """
    Gets scaled crowc image and scaled points for corresponding density map
    :param crowd_image: the crowd image that wanted to be scaled to generate ground truth density map
    :param points: the position set of all points that wanted to be scaled to generate ground truth density map
    :param scale: the scale factor
    :return: sacled crowd image, scaled points
    """
    h = crowd_img.shape[0]
    w = crowd_img.shape[1]
    scaled_crowd_img = cv.resize(crowd_img, (w // scale, h // scale))
    for i in range(len(points)):
        points[i] = points[i] / scale

    return scaled_crowd_img, points


def read_train_data(img_path, gt_path, crop_size=256, scale=8, knn_phase=True, k=2, min_head_size=16, max_head_size=200):
    """
    read_the trianing data from datasets ad the input and label of network
    :param img_path: the crowd image path
    :param gt_path: the label(ground truth) data path
    :param crop_size: the crop size
    :param scale: the scale factor, accorting to the accumulated downsampling factor
    :param knn_phase: True or False, determines wheather to use geometry-adaptive Gaussain kernel or general one
    :param k:  a integer, the number of neareat neighbor
    :param min_head_size: the minimum value of the head size in original crowd image
    :param max_head_size: the maximum value of the head size in original crowd image
    :return: the crwod image as the input of network, the scaled density map as the ground truth of network,
             the ground truth crowd count
    """

    ori_crowd_img = cv.imread(img_path)
    # read the .mat file in dataset
    label_data = loadmat(gt_path)
    points = label_data['image_info'][0][0]['location'][0][0]
    # crowd_count = label_data['image_info'][0][0]['number'][0][0]
    cropped_crowd_img, cropped_points, cropped_crowd_count = get_cropped_crowd_image(ori_crowd_img, points, crop_size=crop_size)

    cropped_scaled_crowd_img, cropped_scaled_points = get_scaled_crowd_image_and_points(cropped_crowd_img, cropped_points, scale=scale)
    # cropped_scaled_crowd_count = cropped_crowd_count
    cropped_scaled_crowd_img_size = [cropped_scaled_crowd_img.shape[0], cropped_scaled_crowd_img.shape[1]]
    scaled_min_head_size = min_head_size / scale
    scaled_max_head_size = max_head_size / scale

    # after cropped and scaled
    density_map = get_density_map(cropped_scaled_crowd_img_size, cropped_scaled_points,
                                  knn_phase, k, scaled_min_head_size, scaled_max_head_size)

    # cropped_crowd_img = np.asarray(cropped_crowd_img)
    cropped_crowd_img = cropped_crowd_img.reshape((1, cropped_crowd_img.shape[0], cropped_crowd_img.shape[1], cropped_crowd_img.shape[2]))
    cropped_crowd_count = np.asarray(cropped_crowd_count).reshape((1, 1))
    cropped_scaled_density_map = density_map.reshape((1, density_map.shape[0], density_map.shape[1], 1))

    return cropped_crowd_img, cropped_scaled_density_map, cropped_crowd_count


def read_test_data(img_path, gt_path, scale=8, deconv_is_used=False, knn_phase=True, k=2, min_head_size=16, max_head_size=200):
    """
    read_the testing data from datasets ad the input and label of network
    :param img_path: the crowd image path
    :param gt_path: the label(ground truth) data path
    :param scale: the scale factor, accorting to the accumulated downsampling factor
    :param knn_phase: True or False, determines wheather to use geometry-adaptive Gaussain kernel or general one
    :param k:  a integer, the number of neareat neighbor
    :param min_head_size: the minimum value of the head size in original crowd image
    :param max_head_size: the maximum value of the head size in original crowd image
    :return: the crwod image as the input of network, the scaled density map as the ground truth of network,
             the ground truth crowd count
    """

    ori_crowd_img = cv.imread(img_path)

    # read the .mat file in dataset
    label_data = loadmat(gt_path)
    points = label_data['image_info'][0][0]['location'][0][0]
    crowd_count = label_data['image_info'][0][0]['number'][0][0]
    h, w = ori_crowd_img.shape[0], ori_crowd_img.shape[1]

    if deconv_is_used:
        h_ = h - (h // scale) % 2
        rh = h_ / h
        w_ = w - (w // scale) % 2
        rw = w_ / w
        ori_crowd_img = cv.resize(ori_crowd_img, (w_, h_))
        points[:, 1] = points[:, 1] * rh
        points[:, 0] = points[:, 0] * rw
    # scaled_crowd_img, scaled_points = ori_crowd_img,points
    scaled_crowd_img, scaled_points = get_scaled_crowd_image_and_points(ori_crowd_img, points, scale=scale)
    # scaled_crowd_count = crowd_count
    scaled_crowd_img_size = [scaled_crowd_img.shape[0], scaled_crowd_img.shape[1]]
    scaled_min_head_size = min_head_size / scale
    scaled_max_head_size = max_head_size / scale

    # after cropped and scaled
    density_map = get_density_map(scaled_crowd_img_size, scaled_points, knn_phase, k, scaled_min_head_size, scaled_max_head_size)
    ori_crowd_img = ori_crowd_img.reshape((1, ori_crowd_img.shape[0], ori_crowd_img.shape[1], ori_crowd_img.shape[2]))
    crowd_count = np.asarray(crowd_count).reshape((1, 1))
    scaled_density_map = density_map.reshape((1, density_map.shape[0], density_map.shape[1], 1))

    return ori_crowd_img, scaled_density_map, crowd_count


def show_density_map(density_map):
    """
    show the density map to help us analysis the distribution of the crowd
    :param density_map: the density map, the shape is [h, w]
    """

    plt.imshow(density_map, cmap='jet')
    plt.show()

'''
a,b,c = get_data_list("G:\ShanghaiTech\part_A")
print(a[1])
# Load the image and ground truth
img, gt_dmp, gt_count = read_train_data(a[1],b[1])

sum = np.sum(np.sum(gt_dmp))

print(sum, gt_count)


dataset = 'A'
# training dataset
img_root_dir = r'G:/ShanghaiTech/part_' + dataset + r'/train_data/images/'
gt_root_dir = r'G:/ShanghaiTech/part_' + dataset + r'/train_data/ground-truth/'
# testing dataset
val_img_root_dir = r'G:/ShanghaiTech/part_' + dataset + r'/test_data/images/'
val_gt_root_dir = r'G:/ShanghaiTech/part_' + dataset + r'/test_data/ground-truth/'

# training dataset file list
img_file_list = os.listdir(img_root_dir)
gt_img_file_list = os.listdir(gt_root_dir)

# testing dataset file list
val_img_file_list = os.listdir(val_img_root_dir)
val_gt_file_list = os.listdir(val_gt_root_dir)

print(img_file_list)
img_path = img_root_dir + img_file_list[0]
gt_path = gt_root_dir + gt_img_file_list[0]

img, gt_dmp, gt_count = read_train_data(img_path, gt_path, scale=4)
print(gt_count)

train_image_list, train_gt_list, iteration = get_data_list('A',mode='train')
img,gt_dmp,gt_count = read_train_data(train_image_list[0],train_gt_list[0],scale=4)
print(gt_count)
'''


