import tensorflow as tf
from tensorflow.contrib import slim
from scipy import misc
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
def get_data_list(data_root, mode='train'):

    if mode == 'train':
        imagepath = os.path.join(data_root, 'train_data', 'images')
        gtpath = os.path.join(data_root, 'train_data', 'ground-truth')

    elif mode == 'valid':
        imagepath = os.path.join(data_root, 'valid_data', 'images')
        gtpath = os.path.join(data_root, 'valid_data', 'ground-truth')

    else:
        imagepath = os.path.join(data_root, 'test_data', 'images')
        gtpath = os.path.join(data_root, 'test_data', 'ground-truth')

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

def reshape_tensor(tensor):
    """
    Reshapes the input tensor appropriate to the network input
    i.e. [1, tensor.shape[0], tensor.shape[1], 1]
    :param tensor: input tensor
    :return: reshaped tensor
    """
    r_tensor = np.reshape(tensor, newshape=(1, tensor.shape[0], tensor.shape[1], 1))
    return r_tensor


import matplotlib.image as mpimg
import scipy.io
a,b,c = get_data_list("G:\ShanghaiTech\part_A")


# Load the image and ground truth
train_image = misc.imread(a[1])
train_d_map = scipy.io.loadmat(b[1])['image_info']

train_image_r = np.reshape(train_image, newshape=(1, train_image.shape[0], train_image.shape[1], 3))
train_d_map_r = np.reshape(train_d_map, newshape=(1, train_d_map.shape[0], train_d_map.shape[1], 1))

print(train_image_r.shape)
print(train_d_map_r.shape)

