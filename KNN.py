import numpy as np
import operator

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T  # []里说明eye中1偏离的位子
    return Y
import numpy as np
import operator

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T  # []里说明eye中1偏离的位子
    return Y


##将cifar-10数据导入
filepath = "/home/ronan/文档/train data/cifar-10-batches-py/"
dict1 = unpickle(filepath + 'data_batch_1')
dict2 = unpickle(filepath + 'data_batch_2')
dict3 = unpickle(filepath + 'data_batch_3')
dict4 = unpickle(filepath + 'data_batch_4')
dict5 = unpickle(filepath + 'data_batch_5')
dict0 = unpickle(filepath + 'test_batch')
x_train = np.vstack((dict1[b'data'], dict2[b'data'], dict3[b'data'], dict4[b'data'], dict5[b'data']))
x_train = x_train.reshape((50000, 3, 32, 32))
Xtr = x_train.transpose(0, 2, 3, 1)

Ytr = np.hstack((dict1[b'labels'], dict2[b'labels'], dict3[b'labels'], dict4[b'labels'], dict5[b'labels']))

x_test = dict0[b'data']
x_test = x_test.reshape((10000, 3, 32, 32))
Xte = x_test.transpose(0, 2, 3, 1)[0:30]

Yte = np.array(dict0[b'labels'])[0:30]
print(Ytr.shape)
print(Yte.shape)


#将数据集展开
Xtr_row = Xtr.reshape(Xtr.shape[0], 32*32*3) # 50000 x 3072
Xte_row = Xte.reshape(Xte.shape[0], 32*32*3)# 50000 x 3072
#构造KNN模型
class NearestNeibor(object):
    def __init__(self):
        pass
    def train(self, x, y):
        self.Xtr = x
        self.Ytr = y
    
    def predict(self, x, k):
        num_test = x.shape[0]
        Ypred = np.zeros(num_test, dtype = self.Ytr.dtype)
        
        for i in range(num_test):
            distances = np.sqrt(np.sum(np.square(self.Xtr - x[i, :]), axis = 1))
            sorteddistances = np.argsort(distances)
            classCount = {}
            for j in range(k):
                currentLable = Ytr[sorteddistances[j]]
                classCount[currentLable] = classCount.get(currentLable, 0) + 1
            sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
            Ypred[i] = sortedClassCount[0][0]

        return Ypred


#预测
model = NearestNeibor()
model.train(Xtr_row, Ytr)
Ypred = model.predict(Xte_row, 5)
print(Ypred)
print(Yte)
# print('accuracy: %f' % (np.mean(Ypred == Yte))
accuracy = np.mean(Ypred == Yte)
print(accuracy)
