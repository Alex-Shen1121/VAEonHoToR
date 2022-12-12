import csv
import random

import numpy as np
import argparse
from tqdm import tqdm


class HoToR_mod:
    def __init__(self):
        self.TestData = dict()
        self.indexItemTrainPurchase = None
        self.indexUserTrainPurchase = None
        self.indexItemTrainClick = None
        self.indexUserTrainClick = None
        self.TrainData = dict()
        self.ItemTrainingSet = set()
        self.num_train_notFive = 0
        self.num_train = 0
        self.itemRatingNumTrain = None
        self.userRatingNumTrain = None
        self.args = None

        self.biasV = None
        self.V = None
        self.U = None

    def readConfigurations(self):
        # 读取实验参数
        parser = argparse.ArgumentParser(description='Implementation For HoToR')
        parser.add_argument('--d', type=int, default=100)
        parser.add_argument('--alpha_u', type=float, default=1e-3)
        parser.add_argument('--alpha_v', type=float, default=1e-3)
        parser.add_argument('--beta_v', type=int, default=1e-3)
        parser.add_argument('--gamma', type=int, default=1e-2)
        parser.add_argument('--n', type=int, default=71567)
        parser.add_argument('--m', type=int, default=10681)
        parser.add_argument('--num_iterations', type=int, default=1000)
        parser.add_argument('--topK', type=int, default=5)
        parser.add_argument('--fnTrainData', type=str, default='../dataset/ML10M-ExplicitPositive4Ranking/ML10M'
                                                               '.ExplicitPositive4Ranking.copy1.explicit')
        parser.add_argument('--fnTestData', type=str, default='../dataset/ML10M-ExplicitPositive4Ranking/ML10M'
                                                              '.ExplicitPositive4Ranking.copy1.test')
        parser.add_argument('--fnEvaluationResult', type=str,
                            default='HoToR_mod_ML10M_TEST_copy1_lambda040_0001_1000.txt')
        parser.add_argument('--lambda_mod', type=int, default=0.4)
        self.args = parser.parse_args()

        print('===================== 读取参数 ====================')
        print("d: " + str(self.args.d))
        print("alpha_u: " + str(self.args.alpha_u))
        print("alpha_v: " + str(self.args.alpha_v))
        print("beta_v: " + str(self.args.beta_v))
        print("gamma: " + str(self.args.gamma))
        print("lambda_mod: " + str(self.args.lambda_mod))
        print("fnTrainData: " + self.args.fnTrainData)
        print("fnTestData: " + self.args.fnTestData)
        print("fnEvaluationResult:" + self.args.fnEvaluationResult)
        print("n: " + str(self.args.n))
        print("m: " + str(self.args.m))
        print("num_iterations: " + str(self.args.num_iterations))
        print("topK: " + str(self.args.topK))

        self.U = np.zeros((self.args.n + 1, self.args.d))
        self.V = np.zeros((self.args.m + 1, self.args.d))
        self.biasV = np.zeros(self.args.m + 1)

    def readData(self):
        print("==================== 正在读取数据 ====================")
        self.userRatingNumTrain = np.zeros(self.args.n + 1)
        self.itemRatingNumTrain = np.zeros(self.args.m + 1)

        # 读取训练集
        with open(self.args.fnTrainData) as csvfile:
            reader = csv.reader(csvfile, delimiter=' ')

            for row in tqdm(reader):
                # 读取数据
                userID = int(row[0])
                itemID = int(row[1])
                rating = float(row[2])
                # print(userID, itemID, rating)

                # 统计每个用户、物品的个数
                self.userRatingNumTrain[userID] += 1
                self.itemRatingNumTrain[itemID] += 1

                # 总训练集大小
                self.num_train += 1
                # 评分非5分 训练集数量
                if rating < 5.0:
                    self.num_train_notFive += 1

                self.ItemTrainingSet.add(itemID)

                # 统计训练集 数据结构如下：
                # TrainData = {userID1: {itemID1:rating1, itemID2:rating2}, userId2: {...}, ...}
                if userID in self.TrainData.keys():
                    itemRatingSet = self.TrainData[userID]
                else:
                    itemRatingSet = {}
                itemRatingSet[itemID] = rating
                self.TrainData[userID] = itemRatingSet

        # 记录用户浏览未购买的记录（小于5分）
        self.indexUserTrainClick = np.zeros(self.num_train_notFive)
        self.indexItemTrainClick = np.zeros(self.num_train_notFive)
        # 记录用户浏览且购买的记录（等于5分）
        self.indexUserTrainPurchase = np.zeros(self.num_train - self.num_train_notFive)
        self.indexItemTrainPurchase = np.zeros(self.num_train - self.num_train_notFive)

        # 统计每个用户/物品的浏览/购买记录
        # 注：user和item列表一一对应
        idx, idx5 = 0, 0
        for u in tqdm(range(1, self.args.n + 1)):
            if u not in self.TrainData:
                continue
            itemSet_u = self.TrainData[u]
            for i in itemSet_u:
                if itemSet_u[i] < 5.0:
                    self.indexUserTrainClick[idx] = u
                    self.indexItemTrainClick[idx] = i
                    idx += 1
                else:
                    self.indexUserTrainPurchase[idx5] = u
                    self.indexItemTrainPurchase[idx5] = i
                    idx5 += 1

        # 读取测试集
        with open(self.args.fnTestData) as csvfile:
            reader = csv.reader(csvfile, delimiter=' ')

            for row in tqdm(reader):
                # 读取数据
                userID = int(row[0])
                itemID = int(row[1])

                # 统计测试集 数据结构如下：
                # TestData = {userID1: set(itemID1, itemID2, ....), userId2: set(...)}
                if userID in self.TestData.keys():
                    itemSet = self.TestData[userID]
                else:
                    itemSet = set()
                itemSet.add(itemID)
                self.TrainData[userID] = itemSet

    def initialization(self):
        print("==================== 统计量初始化 ====================")
        # -- g_avg
        g_avg = 0

        for i in range(1, self.args.m + 1):
            g_avg += self.itemRatingNumTrain[i]
        g_avg = g_avg / self.args.n / self.args.m
        # print("The global average rating:" + str(g_avg))

        # -- biasV
        for i in range(1, self.args.m + 1):
            self.biasV[i] = self.itemRatingNumTrain[i] / self.args.n - g_avg

        # -- V
        for i in range(1, self.args.m + 1):
            for k in range(self.args.d):
                self.V[i][k] = (random.random() - 0.5) * 0.01

        # -- U
        for i in range(1, self.args.n + 1):
            for k in range(self.args.d):
                self.U[i][k] = (random.random() - 0.5) * 0.01

        pass

    def HoToR_mod_training(self):
        print("==================== 正在读取数据 ====================")
        pass

    def test(self):
        print("==================== 正在读取数据 ====================")
        pass

    def main(self):
        self.readConfigurations()
        self.readData()
        self.initialization()


if __name__ == '__main__':
    model = HoToR_mod()
    model.main()
