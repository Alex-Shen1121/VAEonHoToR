import numpy as np
import argparse


class HoToR_mod:
    def __init__(self):
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
        parser.add_argument('--fnTrainData', type=str, default='ML10M.ExplicitPositive4Ranking.copy1.explicit')
        parser.add_argument('--fnTestData', type=str, default='ML10M.ExplicitPositive4Ranking.copy1.test')
        parser.add_argument('--fnEvaluationResult', type=str,
                            default='HoToR_mod_ML10M_TEST_copy1_lambda040_0001_1000.txt')
        parser.add_argument('--lambda_mod', type=int, default=0.4)
        self.args = parser.parse_args()

        # 变量初始化
        self.d = 0
        self.alpha_u = 0.0
        self.alpha_v = 0.0
        self.beta_v = 0.0

        self.gamma = 0.0
        self.topK = 0
        self.lambda_mod = 0.0

        self.n = 0
        self.m = 0
        self.num_train = 0
        self.num_train_notFive = 0
        self.num_iterations = 0

        #  模型参数
        self.U = []
        self.V = []
        self.biasV = []

        self.TrainData = {}
        self.TestData = {}
        self.ItemTrainingSet = set()

        #  File name
        self.fnTrainData = ""
        self.fnTestData = ""
        self.fnEvaluationResult = ""

        # == train data for sampling (ratings=5.0)
        self.indexUserTrainPurchase = []
        self.indexItemTrainPurchase = []

        # == train data for sampling (ratings<5.0)
        self.indexUserTrainClick = []
        self.indexItemTrainClick = []

        self.userRatingNumTrain = []
        self.itemRatingNumTrain = []

    def readConfigurations(self):
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
        pass

    def initialization(self):
        pass

    def HoToR_mod_training(self):
        pass

    def test(self):
        pass


if __name__ == '__main__':
    model = HoToR_mod()
    model.readConfigurations()
