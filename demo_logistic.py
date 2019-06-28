import NeuralNetwork as NN
import numpy as np
import matplotlib.pyplot as plt
import tools

def train(save_model_path):
    # 制作数据集
    train_datas = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    train_targets = np.array([[1],[0],[0],[1]])
    print("The total numbers of datas : ", len(train_datas))

    # 设置训练所需的超参数
    batch_size = 4
    # 训练次数
    train_epochs = 1000
    # 学习率
    lr = 0.01
    decay = False
    regularization = False
    input_features_numbers = train_datas.shape[1]
    layer_structure = [input_features_numbers, 16, 8, 1]
    display = True
    net_name = 'nn'
    # 定义我们的神经网络分类器
    net = NN.MLP(name=net_name, layer_structure=layer_structure, task_model='logistic', batch_size=batch_size)
    # 开始训练
    print("---------开始训练---------")
    net.train(train_datas=train_datas, train_targets=train_targets, train_epoch=train_epochs, lr=lr, lr_decay=decay, loss='BE', regularization=regularization, display=display)
    # 保存模型
    net.save_model(path=save_model_path)
    # 绘制网络的训练损失和精度
    total_net_loss = [net.total_loss]
    total_net_accu = [net.total_accuracy]
    tools.drawDataCurve(total_net_loss, total_net_accu)

def test(save_model_path):
    test_datas = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    test_targets = np.array([[1],[0],[0],[1]])
    print("The total numbers of datas : ", len(test_datas))

    # 设置训练所需的超参数
    batch_size = 4
    input_features_numbers = test_datas.shape[1]
    layer_structure = [input_features_numbers, 16, 8, 1]
    net_name = 'nn'

    # 测试代码
    print("---------测试---------")
    # 载入训练好的模型
    net = NN.MLP(name=net_name, layer_structure=layer_structure, task_model='logistic', batch_size=batch_size, load_model=save_model_path)

    # 网络进行预测
    test_steps = test_datas.shape[0] // batch_size
    accu = 0
    for i in range(test_steps):
        input_data = test_datas[batch_size*i : batch_size*(i+1), :]
        pred = (net(input_data))
        accu += np.sum((pred>0.6) == test_targets) / test_targets.shape[0]
    print("网络识别的准确率 : ", accu / test_steps)

if __name__ == "__main__":
    save_model_path = 'model_log/'
    use_norm = False
    train(save_model_path)
    test(save_model_path)

    
