import numpy as np
import time

def loadData(fileName):
    print('start to read data')
    dataArr = []; labelArr = [] #定义数据和标签集
    fr = open(fileName, 'r')  #读取文件 .readlines()方法是一次按行读取
    for line in fr.readlines():
        curline = line.strip().split(',')  #对每一行去空格然后按照','逗号分割

        if int(curline[0]) >= 5: #大于5的样本顶为正样例
            labelArr.append(1)
        else:
            labelArr.append(-1) #不是话就是负的样例
        dataArr.append([int(num)/255 for num in curline[1:]])  #数据是先进行归一化然后添进数据集

    return dataArr, labelArr

def perceptron(dataArr, labelArr, iter=50): #感知机
    print('start to trans')
    dataMat = np.mat(dataArr)   #把列表换成矩阵
    labelMat = np.mat(labelArr).T  #求标签列表的转置
    m, n = np.shape(dataMat)    #求一下这个矩阵的长宽,其中m是图片的总数(也就是行数),而n是每个图片的大小(被平铺了)
    w = np.zeros((1,np.shape(dataMat)[1])) #多重感知机的权重=输入单元数,且初始化是0
    b = 0
    h = 0.0001

    for k in range(iter):  #对于每一次迭代
        for i in range(m): #对于每一个图片
            xi = dataMat[i] #这个图片的数据和标签分别是 xi和yi
            yi = labelMat[i]
            if -1 * yi * (w * xi.T + b) >= 0: #如果预测的值和标签值不符合
                w = w + h * yi * xi           #我们更新一次权值,更新的方向是向着错误点前进,梯度
                b = b + h * yi                #更新偏移

        print('Round %d:%d training' % (k, iter))

    return w, b
def test(dataArr, labelArr, w, b):
    print('start to test')
    dataMat = np.mat(dataArr)
    labelMat = np.mat(labelArr).T
    m, n =np.shape(dataMat)
    errorCnt = 0         #计算分类错误的图片数量
    for i in range(m):
        xi = dataMat[i]
        yi = labelMat[i]
        result = -1 * yi * (w * xi.T + b) #把已经算好的权值带入进去计算出预测的分类情况
        if result >= 0 : errorCnt += 1    #如果正确的话应该是小于0的,错了计数器就加一

    accruRate = 1 - (errorCnt / m)     #计算正确率
    return accruRate

if __name__ == '__main__':
    start = time.time()   #计时功能
    trainData, trainLabel = loadData('/home/lqd/Downloads/mnist_dataset_csv/mnist_train.csv')
    testData, testLabel = loadData('/home/lqd/Downloads/mnist_dataset_csv/mnist_test.csv')

    w, b = perceptron(trainData, trainLabel,iter=60)
    accruRate = test(testData, testLabel, w, b)
    end = time.time()
    print('accruacy rate is',accruRate)
    print('time span:',end - start)
