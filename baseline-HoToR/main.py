import numpy as np
import argparse

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
parser.add_argument('--fnEvaluationResult', type=str, default='HoToR_mod_ML10M_TEST_copy1_lambda040_0001_1000.txt')
parser.add_argument('--lambda_mod', type=int, default=0.4)
args = parser.parse_args()

U = np.zeros((args.n + 1, args.d))
V = np.zeros((args.m + 1, args.d))
biasV = np.zeros(args.m + 1)


def readConfigurations(args):
    print('===================== 读取参数 ====================')
    print("d: " + str(args.d))
    print("alpha_u: " + str(args.alpha_u))
    print("alpha_v: " + str(args.alpha_v))
    print("beta_v: " + str(args.beta_v))
    print("gamma: " + str(args.gamma))
    print("lambda_mod: " + str(args.lambda_mod))
    print("fnTrainData: " + args.fnTrainData)
    print("fnTestData: " + args.fnTestData)
    print("fnEvaluationResult:" + args.fnEvaluationResult)
    print("n: " + str(args.n))
    print("m: " + str(args.m))
    print("num_iterations: " + str(args.num_iterations))
    print("topK: " + str(args.topK))


def readData():
    pass


def initialization():
    pass


def HoToR_mod_training():
    pass


def test():
    pass


if __name__ == '__main__':

    # 1. read configurations
    readConfigurations(args);

    # 2. read data
    readData()

    # 3. initialization
    initialization()

    # 4. training
    HoToR_mod_training()

    # 5. test
    test()
