import numpy as np
import time

def loadData(filename):
    dataArr = []; labelArr = []
    fr = open(filename)
    for line in fr.readlines():
        curLine = line.strip().split(',')
        dataArr.append([int(int(num)>128) for num in curLine[1:]])
        #上面这行进行了二值化处理，大于128的都改成1了,int(bool)会变成0和1
        labelArr.append(curLine[0])
    return dataArr,labelArr

def NaiveBayes(Py, Px_y, x):
    # Py 先验概率分布  Px_y 条件概率分布  x 要估计的样本x
    featrueNum = 784
    classNum = 10
    P = [0] * classNum
    for i in range(classNum):
        sum = 0
        for j in range(featrueNum):
            sum += Px_y[i][j][x[j]]
        P[i] = sum + Py[i]
        # 朴素贝叶斯最后的计算公式就是上面这个式子
    return P.index(max(P))

def test(Px, Px_y, testDataArr, testLabelArr):
    errorCnt = 0
    for i in range(len(testDataArr)):
        presict = NaiveBayes(Px, Px_y, testDataArr[i])
        if presict != testLabelArr[i]:
            errorCnt += 1
    # 计算估计的正确率
    return 1 - (errorCnt / len(testDataArr))

def getAllProbability(trainDataArr, trainLabelArr):
    featrueNum = 784
    classNum =10

    Py = np.zeros((classNum, 1))
    for i in range(classNum):
        Py[i] = ((np.sum(np.mat(trainLabelArr) == i)) + 1 / (len(trainLabelArr)) + 10)
        # 这里计算的是样本属于中每一类的概率，使用了拉普拉斯平滑(laplace smoothing),防止某一类的概率
        # 是0，这样之后计算条件概率连成的时候不会全是0，而且保证这个值很小。P(Y=ck)
    Py = np.log(Py)
    #为了防止数据下溢,取个log,这在计算似然函数的时候很有用.

    Px_y = np.zeros((classNum, featrueNum, 2))
    # 这里最后一维是2的意思是统计数据集中的0,1的个数，记住我们之前把数据二值化了，所以只会有0,1两种数据
    # 存储之后的代表的每一个类，每一个特征0,1的数量。
    for i in range(len(trainLabelArr)):
        label = trainLabelArr[i]
        x = trainDataArr[i]
        for j in range(featrueNum):
            Px_y[label][j][x[j]] += 1

    # 这里的循环把之前存的0,1的个数变成了条件概率
    for label in range(classNum):
        for j in range(featrueNum):
            Px_y0 = Px_y[label][j][0] #这个代表的是这个label标签下，j特征为0的个数
            Px_y1 = Px_y[label][j][1] #这个代表的是这个label标签下，j特征为1的个数
            Px_y[label][j][0] = np.log((Px_y0 + 1) / (Px_y0 + Px_y1 +2))
            # 这里更新了一遍矩阵中的值，变成了条件概率了
            Px_y[label][j][1] = np.log((Px_y1 + 1) / (Px_y0 + Px_y1 +2))
    return Py, Px_y

if __name__ == "__main__":
    start = time.time()
    print('start read trainSet')
    trainDataArr, trainLabelArr = loadData()
    print('start read testSet')
    testDataArr, testLabelArr = loadData()
    print('start to train')
    Py, Px_y = getAllProbability(trainDataArr, trainLabelArr)
    print('start to test')
    accuracy = test(Py, Px_y, testDataArr, testLabelArr)
    print('the accuracy is:', accuracy)
    print('time span:', time.time() - start)
