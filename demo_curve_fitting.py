import NeuralNetwork as NN
import numpy as np
import matplotlib.pyplot as plt
import tools

def train(save_model_path):
    # 制作数据集
    train_datas = np.linspace(0, 2*np.pi, 5000).reshape(5000, 1)
    train_targets = np.sin(train_datas).reshape(5000, 1)
    print("The total numbers of datas : ", len(train_datas))

    # 设置训练所需的超参数
    batch_size = 5000
    # 训练次数
    train_epochs = 1000
    # 学习率
    lr = 0.01
    decay = False
    regularization = False
    input_features_numbers = train_datas.shape[1]
    layer_structure = [input_features_numbers, 16, 8, 4, 1]
    display = True
    net_name = 'nn'
    # 定义我们的神经网络分类器
    net = NN.MLP(name=net_name, layer_structure=layer_structure, task_model='regression', batch_size=batch_size)
    # 开始训练
    print("---------开始训练---------")
    net.train(train_datas=train_datas, train_targets=train_targets, train_epoch=train_epochs, lr=lr, lr_decay=decay, loss='BE', regularization=regularization, display=display)
    # 保存模型
    net.save_model(path=save_model_path)
    # 绘制网络的训练损失和精度
    total_net_loss = [net.total_loss]
    tools.drawDataCurve(total_net_loss)

def test(save_model_path):
    # 读取xlsx文件
    test_datas = np.linspace(0, 2*np.pi, 5000).reshape(5000, 1)
    test_targets = np.sin(test_datas).reshape(5000, 1)
    print("The total numbers of datas : ", len(test_datas))

    # 设置训练所需的超参数
    batch_size = 5000
    input_features_numbers = test_datas.shape[1]
    layer_structure = [input_features_numbers, 16, 8, 4, 1]
    net_name = 'nn'

    # 测试代码
    print("---------测试---------")
    # 载入训练好的模型
    net = NN.MLP(name=net_name, layer_structure=layer_structure, task_model='regression', batch_size=batch_size, load_model=save_model_path)

    # 网络进行预测
    test_steps = test_datas.shape[0] // batch_size
    pred = []
    for i in range(test_steps):
        input_data = test_datas[batch_size*i : batch_size*(i+1), :]
        pred.append(net(input_data))
    pred = np.concatenate(pred, 0)
    plt.plot(test_datas, test_targets)
    plt.plot(test_datas, pred)
    plt.show()

if __name__ == "__main__":
    save_model_path = 'model_reg/'
    use_norm = False
    train(save_model_path)
    test(save_model_path)

    
