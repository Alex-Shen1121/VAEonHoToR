import csv
import heapq
import math
import random

import numpy as np
import argparse
from tqdm import tqdm


class TopKHeap:
    def __init__(self, k):
        self.k = k
        self.data = []

    def push(self, elem):
        if len(self.data) < self.k:
            heapq.heappush(self.data, elem)
        else:
            topk_small = self.data[0]
            if elem > topk_small:
                heapq.heapreplace(self.data, elem)

    def topk(self):
        return [x for x in reversed([heapq.heappop(self.data) for x in range(len(self.data))])]


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
        parser.add_argument('--beta_v', type=float, default=1e-3)
        parser.add_argument('--gamma', type=float, default=1e-2)
        parser.add_argument('--n', type=int, default=71567)
        parser.add_argument('--m', type=int, default=10681)
        parser.add_argument('--num_iterations', type=int, default=1000)
        parser.add_argument('--topK', type=int, default=5)
        parser.add_argument('--fnTrainData', type=str, default='../dataset/ML10M-ExplicitPositive4Ranking/ML10M'
                                                               '.ExplicitPositive4Ranking.copy1.explicit')
        parser.add_argument('--fnTestData', type=str, default='../dataset/ML10M-ExplicitPositive4Ranking/ML10M'
                                                              '.ExplicitPositive4Ranking.copy1.test')
        parser.add_argument('--fnEvaluationResult', type=str, default='../dataset/ML10M-ExplicitPositive4Ranking'
                                                                      '/HoToR_mod_ML10M_TEST_copy1_lambda040_0001_1000.txt')
        parser.add_argument('--lambda_mod', type=float, default=0.4)
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
                self.TestData[userID] = itemSet

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

    def HoToR_mod_training(self):
        print("==================== 正在训练模型 ====================")
        for iter in tqdm(range(self.args.num_iterations)):
            if iter % 50 == 0:
                self.test()
            # if iter % 5 == 0:
            #     print("===================== iter" + str(iter) + " ===================")
            for iter2 in range(self.num_train):

                # 随机采样样本
                randomNum = random.random()
                if randomNum < self.args.lambda_mod:
                    # 从浏览记录中随机采样(u, i, r_ui)数据
                    idx = int(math.floor(random.random() * self.num_train_notFive))
                    u = int(self.indexUserTrainClick[idx])
                    i = int(self.indexItemTrainClick[idx])
                    r_ui = self.TrainData.get(u).get(i)

                    # todo: 不理解为什么
                    if r_ui == 0:
                        r_ui = 3

                    # 计算(u, i)权重
                    barr_ui = (2 ** r_ui - 1) / (2 ** 5)

                    # 随机采样一个物品j（未浏览）
                    tmp = self.TrainData.get(u)
                    while True:
                        j = int(math.floor(random.random() * self.args.m)) + 1
                        if j in self.ItemTrainingSet and j not in tmp.keys():
                            break
                else:
                    # 从购买记录中随机采样(u, i, r_ui)数据 r_ui = 5
                    idx = int(math.floor(random.random() * (self.num_train - self.num_train_notFive)))
                    u = int(self.indexUserTrainPurchase[idx])
                    i = int(self.indexItemTrainPurchase[idx])
                    barr_ui = 1

                    # 随机采样一个物品j（未购买）
                    tmp = self.TrainData.get(u)
                    while True:
                        j = int(math.floor(random.random() * self.args.m)) + 1
                        if j in self.ItemTrainingSet and (j not in tmp.keys() or tmp.get(j) != 5):
                            break

                # 计算损失函数
                r_uij = (np.dot(self.U[u], self.V[i]) + self.biasV[i]) - (np.dot(self.U[u], self.V[j]) + self.biasV[j])
                loss_uij = -1 / (1 + math.exp(r_uij))

                # 更新梯度
                grad_Uuk = loss_uij * (self.V[i] - self.V[j]) + self.args.alpha_u * self.U[u]
                grad_Vik = loss_uij * self.U[u] + self.args.alpha_v * self.V[i]
                grad_Vjk = loss_uij * (-self.U[u]) + self.args.alpha_v * self.V[j]

                # Update parameters using vectorized operations
                self.U[u] = self.U[u] - self.args.gamma * grad_Uuk * barr_ui
                self.V[i] = self.V[i] - self.args.gamma * grad_Vik * barr_ui
                self.V[j] = self.V[j] - self.args.gamma * grad_Vjk * barr_ui

                # Update bias terms
                grad_bi = loss_uij + self.args.beta_v * self.biasV[i]
                grad_bj = loss_uij * (-1) + self.args.beta_v * self.biasV[j]
                self.biasV[i] = self.biasV[i] - self.args.gamma * grad_bi * barr_ui
                self.biasV[j] = self.biasV[j] - self.args.gamma * grad_bj * barr_ui

    def test(self):
        # print("==================== 正在评估数据 ====================")
        DCGbest = [0] * (self.args.topK + 1)
        PrecisionSum = [0] * (self.args.topK + 1)
        RecallSum = [0] * (self.args.topK + 1)
        F1Sum = [0] * (self.args.topK + 1)
        NDCGSum = [0] * (self.args.topK + 1)
        OneCallSum = [0] * (self.args.topK + 1)

        for k in range(1, self.args.topK + 1):
            DCGbest[k] = DCGbest[k - 1]
            DCGbest[k] += 1 / math.log(k + 1)

        UserNum_TestData = 0
        for u in range(1, self.args.n + 1):
            if u not in self.TrainData.keys() or u not in self.TestData.keys():
                continue
            UserNum_TestData += 1

            # 获取用户u的购买记录（测试集）
            ItemSet_u_TestData = self.TestData.get(u)
            ItemNum_u_TestData = len(ItemSet_u_TestData)

            # 对每个物品进行评分预测
            # 取评分预测最高的TopK
            item2Prediction = TopKHeap(self.args.topK)
            for i in range(1, self.args.m + 1):
                if i not in self.ItemTrainingSet or i in self.TrainData.get(u):
                    continue
                pred = np.dot(self.U[u], self.V[i]) + self.biasV[i]

                item2Prediction.push([pred, i])
            res = item2Prediction.topk()
            TopKResult = [x[1] for x in res]

            # 指标评估
            DCG, DCGbest2 = [0] * (self.args.topK + 1), [0] * (self.args.topK + 1)
            HitSum = 0
            for k in range(1, self.args.topK + 1):
                DCG[k] = DCG[k - 1]
                if TopKResult[k - 1] in ItemSet_u_TestData:
                    HitSum += 1
                    DCG[k] += 1 / math.log(k + 1)

                prec = HitSum / k
                rec = HitSum / ItemNum_u_TestData
                F1 = 2 * prec * rec / (prec + rec) \
                    if prec + rec > 0 else 0

                PrecisionSum[k] += prec
                RecallSum[k] += rec
                F1Sum[k] += F1

                DCGbest2[k] = DCGbest[k] \
                    if len(ItemSet_u_TestData) >= k \
                    else DCGbest2[k - 1]

                NDCGSum[k] += DCG[k] / DCGbest2[k]
                OneCallSum[k] += 1 if HitSum > 0 else 0

        for k in range(5, self.args.topK + 1, 5):
            print(f'Prec@{k}: {PrecisionSum[k] / UserNum_TestData}')
        for k in range(5, self.args.topK + 1, 5):
            print(f'Rec@{k}: {RecallSum[k] / UserNum_TestData}')
        for k in range(5, self.args.topK + 1, 5):
            print(f'F1@{k}: {F1Sum[k] / UserNum_TestData}')
        for k in range(5, self.args.topK + 1, 5):
            print(f'NDCG@{k}: {NDCGSum[k] / UserNum_TestData}')
        for k in range(5, self.args.topK + 1, 5):
            print(f'1-call@{k}: {OneCallSum[k] / UserNum_TestData}')

    def main(self):
        self.readConfigurations()
        self.readData()
        self.initialization()
        self.HoToR_mod_training()


if __name__ == '__main__':
    model = HoToR_mod()
    model.main()
