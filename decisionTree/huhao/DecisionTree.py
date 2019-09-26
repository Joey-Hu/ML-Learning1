import operator


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
    pass


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
            prob = len(subDataSet)/float(len(dataSet))





def splitDataSet(dataSet, index, value):
    pass


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
    del(labels[bestFeature])

    # 取出最优类，以它为branch做分类
    featureValues = [example[bestFeature] for example in dataSet]
    uniqueVals = set(featureValues)
    for value in uniqueVals:
        subLabel = labels[:]
        myTree[bestFeatureLabel][value] = createTree(splitDataSet(dataSet, bestFeature, value), subLabel)
    return myTree



    