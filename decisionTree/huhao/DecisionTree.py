import operator
from math import log
import huhao.decisionTreePlot as dtplot


def majorityCnt(classList):
    '''
    选择出现次数最多的一个结果

    :param classList: label列的集合
    :return: bestFeature 最优特征列
    '''
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    # 倒叙排列classCount得到一个字典集合，然后取出第一个就是结果（yes/no），即出现次数最多的结果
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def calcShannonEnt(dataSet):
    '''
    计算给定数据集的熵
    :param dataSet:
    :return: 返回每一组feature下的某个分类下，熵的信息期望
    '''
    numEntries = len(dataSet)

    # 计算分类标签label出现的次数
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1

        shannonEnt = 0.0
        for key in labelCounts:
            prob = float(labelCounts[key]) / numEntries
            shannonEnt -= prob * log(prob, 2)
    return shannonEnt


def chooseBestFeatureToSplit(dataSet):
    '''
    选择切分数据集的最佳特征

    :param dataSet: 需要切分的数据集
    :return: bestFeature -- 切分数据集的最优的特征列
    '''
    # 求第一行有多少列的 Feature, 最后一列是label列嘛
    numFeature = len(dataSet[0]) - 1
    # label的信息熵
    baseEntropy = calcShannonEnt(dataSet)
    # 最优的信息增益值, 和最优的Featurn编号
    bestInfoGain, bestFeature = 0.0, -1

    for i in range(numFeature):
        featList = [example[i] for example in dataSet]
        # get a set of unique values
        # 获取剔重后的集合，使用set对list数据进行去重
        uniqueVals = set(featList)
        newEntropy = 0.0

        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)

        infoGain = baseEntropy - newEntropy
        print('infoGain=', infoGain, 'bestFeature=', i, baseEntropy, newEntropy)

        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature


def splitDataSet(dataSet, index, value):
    '''
    划分数据集
    splitDataSet(通过遍历dataSet数据集，求出index对应的colnum列的值为value的行)
        就是依据index列进行分类，如果index列的数据等于 value的时候，就要将 index 划分到我们创建的新的数据集中

    :param dataSet: 待划分的数据集
    :param index: 划分数据集的特征
    :param value: 需要返回的特征的值
    :return:
    '''
    retDataSet = []
    for featVec in dataSet:
        if featVec[index] == value:
            reducedFeatVec = featVec[:index]
            reducedFeatVec.extend(featVec[index + 1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet


def createTree(dataSet, labels):
    '''
    创建决策树

    :param dataSet: 要创建决策树的训练集
    :param labels: 训练集中特征对应的含义的labels
    :return: 返回一个决策树
    '''
    classList = [example[-1] for example in dataSet]
    # 终止条件1：所有类标签全部相同，直接返回该类标签
    if classList.count(classList[0]) == len(classList):
        return classList[0]

    # 终止条件2：第二个停止条件：使用完了所有特征，仍然不能将数据集划分成仅包含唯一类别的分组。
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)

    # 选择最优的属性作为分类节点
    bestFeature = chooseBestFeatureToSplit(dataSet)
    bestFeatureLabel = labels[bestFeature]

    # 初始化myTree
    myTree = {bestFeatureLabel: {}}
    del (labels[bestFeature])

    # 取出最优类，以它为branch做分类
    featureValues = [example[bestFeature] for example in dataSet]
    uniqueVals = set(featureValues)
    for value in uniqueVals:
        subLabel = labels[:]
        myTree[bestFeatureLabel][value] = createTree(splitDataSet(dataSet, bestFeature, value), subLabel)
    return myTree


def createDataSet():
    # dataSet 前两列是特征，最后一列对应的是每条数据对应的分类标签
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    # dataSet = [['yes'],
    #         ['yes'],
    #         ['no'],
    #         ['no'],
    #         ['no']]
    # labels  露出水面   脚蹼，注意：这里的labels是写的 dataSet 中特征的含义，并不是对应的分类标签或者说目标变量
    labels = ['no surfacing', 'flippers']
    # 返回
    return dataSet, labels


def classify(inputTree, featLabels, testVec):
    """
    Desc:
        对新数据进行分类
    Args:
        inputTree  -- 已经训练好的决策树模型
        featLabels -- Feature标签对应的名称，不是目标变量
        testVec    -- 测试输入的数据
    Returns:
        classLabel -- 分类的结果值，需要映射label才能知道名称
    """
    # 获取tree的根节点对于的key值
    firstStr = list(inputTree.keys())[0]
    # 通过key得到根节点对应的value
    secondDict = inputTree[firstStr]
    # 判断根节点名称获取根节点在label中的先后顺序，这样就知道输入的testVec怎么开始对照树来做分类
    featIndex = featLabels.index(firstStr)
    # 测试数据，找到根节点对应的label位置，也就知道从输入的数据的第几位来开始分类
    key = testVec[featIndex]
    valueOfFeat = secondDict[key]
    print('+++', firstStr, 'xxx', secondDict, '---', key, '>>>', valueOfFeat)
    # 判断分枝是否结束: 判断valueOfFeat是否是dict类型
    if isinstance(valueOfFeat, dict):
        classLabel = classify(valueOfFeat, featLabels, testVec)
    else:
        classLabel = valueOfFeat
    return classLabel


def fishTest():
    '''
    对动物是否是鱼类分类的测试函数，并将结果使用 matplotlib 画出来
    :return: 
    '''
    myDat, labels = createDataSet()
    import copy
    myTree = createTree(myDat, copy.deepcopy(labels))
    print(myTree)
    print(classify(myTree, labels, [1, 1]))

    dtplot.createPlot(myTree)


if __name__ == "__main__":
    fishTest()
