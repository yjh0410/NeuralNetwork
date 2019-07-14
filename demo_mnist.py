import NeuralNetwork as NN
import numpy as np
import matplotlib.pyplot as plt
import tools

def train(path_to_datas, save_model_path):
    # 读取MNIST数据集
    train_datas, labels = tools.load_mnist(path_to_datas, 'train')
    print("The total numbers of datas : ", len(train_datas))
    train_labels = np.zeros((labels.shape[0], 10))
    train_labels[np.arange(labels.shape[0]), labels.astype('int').reshape(-1)-1] = 1.0

    # 设置训练所需的超参数
    batch_size = 100
    # 训练次数
    train_epochs = 10
    # 学习率
    lr = 0.01
    decay = False
    regularization = False
    input_features_numbers = train_datas.shape[1]
    layer_structure = [input_features_numbers, 512, 256, 128, 10]
    display = True
    net_name = 'nn'
    # 定义我们的神经网络分类器
    net = NN.MLP(name=net_name, layer_structure=layer_structure, task_model='multi', batch_size=batch_size)
    # 开始训练
    print("---------开始训练---------")
    net.train(train_datas=train_datas, train_targets=train_labels, train_epoch=train_epochs, lr=lr, lr_decay=decay, loss='BE', regularization=regularization, display=display)
    # 保存模型
    net.save_model(path=save_model_path)
    # 绘制网络的训练损失和精度
    total_net_loss = [net.total_loss]
    total_net_accuracy = [net.total_accuracy]
    tools.drawDataCurve(total_net_loss, total_net_accuracy)

def test(path_to_datas, save_model_path):
    # 读取xlsx文件
    test_datas, all_label = tools.load_mnist(path_to_datas, 'test')
    print("The total numbers of datas : ", len(test_datas))
    test_labels = np.zeros((all_label.shape[0], 10))
    test_labels[np.arange(all_label.shape[0]), all_label.astype('int').reshape(-1)-1] = 1.0

    # 设置训练所需的超参数
    batch_size = 100
    input_features_numbers = test_datas.shape[1]
    layer_structure = [input_features_numbers, 512, 256, 128, 10]
    net_name = 'nn'

    # 测试代码
    print("---------测试---------")
    # 载入训练好的模型
    net = NN.MLP(name=net_name, layer_structure=layer_structure, task_model='multi', batch_size=batch_size, load_model=save_model_path)

    # 网络进行预测
    test_steps = test_datas.shape[0] // batch_size
    accuracy = 0
    for i in range(test_steps):
        input_data = test_datas[batch_size*i : batch_size*(i+1), :].reshape(batch_size, test_datas.shape[1])
        targets = test_labels[batch_size*i : batch_size*(i+1), :].reshape(batch_size, test_labels.shape[1])

        pred = net(input_data)
        # 计算准确率
        accuracy += np.sum(np.argmax(pred,1) == np.argmax(targets,1)) / targets.shape[0]
    print("网络识别的准确率 : ", accuracy / test_steps)

if __name__ == "__main__":
    path_to_datas = 'mnist/'
    save_model_path = 'model/'
    train(path_to_datas, save_model_path)
    test(path_to_datas, save_model_path)
    
