import numpy as np
import matplotlib.pyplot as plt

class MLP():
    def __init__(self, name='nn', layer_structure=[], task_model=None, batch_size=1, load_model=None):
        """layer_number : 神经网络的层数
           layer_structure = [输入的特征个数，第1层神经元个数，第2层神经元个数，...，最后一层神经元个数输出层特征个数]，
           如网络层数设为layer_number=3, layer_structure=[20,10,5,1]：输入特征是20个，第一层有10个神经元，第二层5个，第三层1个.
           output_model = 'regression'/'logistic'
        """
        self.name = name
        self.layer_number = len(layer_structure) - 1
        self.layer_structure = layer_structure
        self.task_model = task_model
        self.W = []
        self.B = []
        self.batch_size = batch_size
        self.total_loss = []
        if self.task_model == 'logistic' or self.task_model == 'multi':
            self.total_accuracy = []
        
        if load_model == None:
            print("Initializing the network from scratch ...")
            for index in range(self.layer_number):
                self.W.append(np.random.randn(self.layer_structure[index], self.layer_structure[index+1]))
                self.B.append(np.random.randn(1, self.layer_structure[index+1]))
        else:
            print("Initializing the network from trained model ...")
            for index in range(self.layer_number):
                self.W.append(np.loadtxt(load_model + self.name + "_layer_" + str(index) + "_W.txt").reshape(self.layer_structure[index], self.layer_structure[index+1]))
                self.B.append(np.loadtxt(load_model + self.name + "_layer_" + str(index) + "_B.txt").reshape(1, self.layer_structure[index+1]))

    def normal_parameters(self, means, sigmas):
        self.means = means
        self.sigams = sigmas

    def sigmoid(self, x):
        return 1/(1+np.exp(-x))

    def sigmoid_gradient(self, x):
        return self.sigmoid(x)*(1-self.sigmoid(x))
	
    def softmax(self, x):
        return np.exp(x)/np.sum(np.exp(x), axis = 1, keepdims = True)
    
    def forward(self, x):
        """
            intput : x = [batch_size, features]
        """
        self.before_activation = []
        self.activations = [x]
        for index in range(self.layer_number):
            if index < self.layer_number - 1:
                Z = np.dot(self.activations[index], self.W[index]) + self.B[index]
                self.before_activation.append(Z)
                self.activations.append(self.sigmoid(Z))
            else:
                if self.task_model == 'logistic':
                    Z = np.dot(self.activations[index], self.W[index]) + self.B[index]
                    self.before_activation.append(Z)
                    self.activations.append(self.sigmoid(Z))
                elif self.task_model == 'regression':
                    Z = np.dot(self.activations[index], self.W[index]) + self.B[index]
                    self.before_activation.append(Z)
                    self.activations.append(Z)
                elif self.task_model == 'multi':
                    Z = np.dot(self.activations[index], self.W[index]) + self.B[index]
                    self.before_activation.append(Z)
                    self.activations.append(self.softmax(Z))

        return self.activations[-1]

    def __call__(self, x):
        return self.forward(x)

    def lossfunction(self, inputs, target):
        if self.task_model == 'regression':
            return(np.mean(np.sum((inputs - target)**2, 1)))
        elif self.task_model == 'logistic':
            return np.mean(np.sum(-target*np.log(inputs+1e-14) - (1-target)*np.log(1-inputs+1e-14), 1))
        elif self.task_model == 'multi':
            return np.mean(np.sum(-target*np.log(inputs+1e-14), 1))

    def back_forward(self, targets=None, loss=None, regularization=False):
        self.dWs = []
        self.dBs = []
        self.dAs = []
        W_reverse = self.W[::-1]
        activations_reverse = self.activations[::-1]
        before_activation_reverse = self.before_activation[::-1]
        # 从最后一层开始往回传播
        for k in range(self.layer_number):
            if(k == 0):
                if loss == 'MSE' or loss == 'CE' or loss == 'BE':
                    dZ = activations_reverse[k] - targets
                    dW = 1/self.batch_size*np.dot(activations_reverse[k+1].T, dZ)
                    dB = 1/self.batch_size*np.sum(dZ, axis = 0, keepdims = True)
                    dA_before = np.dot(dZ, W_reverse[k].T)
                    self.dWs.append(dW)
                    self.dBs.append(dB)
                    self.dAs.append(dA_before)
            else:
                dZ = self.dAs[k-1]*self.sigmoid_gradient(before_activation_reverse[k])
                dW = 1/self.batch_size*np.dot(activations_reverse[k+1].T,dZ)
                dB = 1/self.batch_size*np.sum(dZ, axis = 0, keepdims = True)
                dA_before = np.dot(dZ, W_reverse[k].T)
                self.dWs.append(dW)
                self.dBs.append(dB)
                self.dAs.append(dA_before)
        self.dWs = self.dWs[::-1]
        self.dBs = self.dBs[::-1]
        
    def steps(self, lr=0.001, lr_decay=False):
        for index in range(len(self.dWs)):
            self.W[index] -= lr*self.dWs[index]
            self.B[index] -= lr*self.dBs[index]

    def train(self, train_datas=None, train_targets=None, train_epoch=1, lr=0.001, lr_decay=False, loss='MSE', regularization=False, display=False):
        train_counts = 0
        for epoch in range(train_epoch):
            if epoch == int(train_epoch * 0.7) and lr_decay == True:
                lr *= 0.1
            train_steps = train_datas.shape[0] // self.batch_size
            for i in range(train_steps):
                input_data = train_datas[self.batch_size*i : self.batch_size*(i+1), :].reshape(self.batch_size, train_datas.shape[1])
                targets = train_targets[self.batch_size*i : self.batch_size*(i+1), :].reshape(self.batch_size, train_targets.shape[1])
                prediction = self.forward(input_data)
                forward_loss = self.lossfunction(prediction, targets)
                if self.task_model=='logistic':
                    accuracy = np.sum((prediction>0.6) == targets) / targets.shape[0]
                    self.total_accuracy.append(accuracy)
                elif self.task_model=='multi':
                    accuracy = np.sum(np.argmax(prediction,1) == np.argmax(targets,1)) / targets.shape[0]
                    self.total_accuracy.append(accuracy)                    
                self.total_loss.append(forward_loss)
                if display:
                    if train_counts % 10 == 0:
                        if self.task_model == 'logistic' or self.task_model == 'multi':
                            print("After " + str(train_counts) + ", loss is ", forward_loss,
                            ", accuracy is ", accuracy)
                        else:
                            print("After " + str(train_counts) + ", loss is ", forward_loss)
                self.back_forward(targets=targets, loss=loss, regularization=regularization)
                self.steps(lr=lr, lr_decay=lr_decay)
                train_counts += 1

    def save_model(self, path):
        print("Saving the " + self.name + " model ...")
        for i in range(self.layer_number):
            np.savetxt(path  + self.name + "_layer_" + str(i) + "_W.txt", self.W[i])
            np.savetxt(path  + self.name + "_layer_" + str(i) + "_B.txt", self.B[i])
        print("Model saved !!!")

                