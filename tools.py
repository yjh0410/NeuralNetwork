import numpy as np
import matplotlib.pyplot as plt
import os
import struct

def load_mnist(path, data_type='train'):
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels.idx1-ubyte'
                               % data_type)
    images_path = os.path.join(path,
                               '%s-images.idx3-ubyte'
                               % data_type)
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II',
                                 lbpath.read(8))
        labels = np.fromfile(lbpath,
                             dtype=np.uint8)
        labels = labels.reshape(labels.shape[0], 1)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII',
                                               imgpath.read(16))
        images = np.fromfile(imgpath,
                             dtype=np.uint8).reshape(len(labels), 784)

    return images, labels

def normalization(data, means=None, sigmas=None):
    """标准差的归一化"""
    if isinstance(means, np.ndarray) and isinstance(means, np.ndarray):
        data = (data - means) / sigmas
        return data
    else:
        # 计算每个特征的均值
        means = np.mean(data, 0)
        # 计算每个特征的标准差
        sigmas = np.std(data, 0)
        # 数据标准化
        data = (data - means) / sigmas
        return data, means, sigmas

def drawDataCurve(loss=[], accu=[]):
    total = len(loss)
    if total == 1:
        width = 1
    else:
        width = total // 2
    height = total // width
    count = 0
    if loss:
        for count in range(total):
            plt.subplot(height, width, count+1)
            plt.plot(loss[count])
            plt.xlabel('training steps')
            plt.ylabel('loss')
            plt.title('training loss'+str(count+1))
        plt.show()
    if accu:
        for count in range(total):
            plt.subplot(height, width, count+1)
            plt.plot(accu[count])
            plt.xlabel('training steps')
            plt.ylabel('accuracy')
            plt.title('training accuracy'+str(count+1))
        plt.show()