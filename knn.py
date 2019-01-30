import numpy as np
import time

def loadData(filename):
    print('start load data')
    dataArr = []; labelArr = []
    fr = open(filename)
    for line in fr.readlines():
        curline = line.strip().split(',')
        dataArr.append([int(num) for num in curline[1:]])
        labelArr.append(int(curline[0]))

    return dataArr, labelArr

def calcDist(x1, x2):
    return np.linalg.norm(x1-x2) #np.sqrt(np.sum(np.square(x1 - x2)))

def getClosest(trainDataMat, trainLabelMat, x, topK):
    distList = [0] * len(trainDataMat)
    for i in range(len(trainDataMat)):
        x1 = trainDataMat[i]
        curDist = calcDist(x1, x)
        distList[i] = curDist

    topKList = np.argsort(np.array(distList))[:topK]
    labelList = [0] * 10
    for index in topKList: #根据标签的名称给所属于的类别打分
        labelList[int(trainLabelMat[index])] += 1
    return labelList.index(max(labelList))

def test(trainDataArr, trainLabelArr, testDataArr, testLabelArr, topK):
    print('start test')
    trainDataMat = np.mat(trainDataArr); trainLabelMat = np.mat(trainLabelArr).T
    testDataMat = np.mat(testDataArr); testLabelMat = np.mat(testLabelArr).T

    errorCnt = 0
    for i in range(200):
        print('test %d:%d' % (i, 200))
        x = testDataMat[i]
        y = getClosest(trainDataMat, trainLabelMat, x, topK)
        if y != testLabelMat[i]: errorCnt +=1

    return 1 - (errorCnt / 200)

if __name__ == '__main__':
    start = time.time()
    trainDataArr, trainLabelArr = loadData('/home/lqd/Downloads/mnist_dataset_csv/mnist_train.csv')
    testDataArr, testLabelArr = loadData('/home/lqd/Downloads/mnist_dataset_csv/mnist_test.csv')
    accur = test(trainDataArr, trainLabelArr, testDataArr, testLabelArr, 25)
    print('accur is: %d' % (accur * 100), '%')
    end = time.time()
    print('time span:',end - start)
