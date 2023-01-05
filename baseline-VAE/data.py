import os
import pandas as pd
from scipy import sparse
import numpy as np


class DataLoader():
    '''
    Load Movielens-20m dataset
    '''

    def __init__(self, path):
        self.pro_dir = os.path.join(path, 'pro_sg')
        assert os.path.exists(self.pro_dir), "Preprocessed files does not exist. Run data.py"

        self.n_items = self.load_n_items()

    def load_data(self, datatype='train'):
        if datatype == 'train':
            return self._load_train_data()
        elif datatype == 'validation':
            return self._load_tr_te_data(datatype)
        elif datatype == 'test':
            return self._load_tr_te_data(datatype)
        else:
            raise ValueError("datatype should be in [train, validation, test]")

    def load_n_items(self):
        unique_sid = list()
        with open(os.path.join(self.pro_dir, 'unique_sid.txt'), 'r') as f:
            for line in f:
                unique_sid.append(line.strip())
        n_items = len(unique_sid)
        return n_items

    def _load_train_data(self):
        path = os.path.join(self.pro_dir, 'train.csv')

        tp = pd.read_csv(path)
        n_users = tp['uid'].max() + 1

        rows, cols = tp['uid'], tp['sid']
        data = sparse.csr_matrix((np.ones_like(rows),
                                  (rows, cols)), dtype='float64',
                                 shape=(n_users, self.n_items))
        return data

    def _load_tr_te_data(self, datatype='test'):
        tr_path = os.path.join(self.pro_dir, '{}_tr.csv'.format(datatype))
        te_path = os.path.join(self.pro_dir, '{}_te.csv'.format(datatype))

        tp_tr = pd.read_csv(tr_path)
        tp_te = pd.read_csv(te_path)

        start_idx = min(tp_tr['uid'].min(), tp_te['uid'].min())
        end_idx = max(tp_tr['uid'].max(), tp_te['uid'].max())

        rows_tr, cols_tr = tp_tr['uid'] - start_idx, tp_tr['sid']
        rows_te, cols_te = tp_te['uid'] - start_idx, tp_te['sid']

        data_tr = sparse.csr_matrix((np.ones_like(rows_tr),
                                     (rows_tr, cols_tr)), dtype='float64',
                                    shape=(end_idx - start_idx + 1, self.n_items))
        data_te = sparse.csr_matrix((np.ones_like(rows_te),
                                     (rows_te, cols_te)), dtype='float64',
                                    shape=(end_idx - start_idx + 1, self.n_items))
        return data_tr, data_te


def get_count(tp, id):
    playcount_groupbyid = tp[[id]].groupby(id, as_index=False)
    count = playcount_groupbyid.size()
    return count


def filter_triplets(tp, min_uc=5, min_sc=0):
    if min_sc > 0:
        itemcount = get_count(tp, 'movieId')
        # todo: 不知道有什么用
        # tp = tp[tp['movieId'].isin(itemcount.index[itemcount['size'] >= min_sc])]

    if min_uc > 0:
        # 根据userId进行groupby，筛选出userID-size（userid有size个4分以上评分记录）
        usercount = get_count(tp, 'userId')
        # todo: 感觉写的不太对
        # tp = tp[tp['userId'].isin(usercount.index[usercount['size'] >= min_uc])]

    usercount, itemcount = get_count(tp, 'userId'), get_count(tp, 'movieId')
    return tp, usercount, itemcount


def split_train_test_proportion(data, test_prop=0.2):
    data_grouped_by_user = data.groupby('userId')
    tr_list, te_list = list(), list()

    np.random.seed(98765)

    for _, group in data_grouped_by_user:
        n_items_u = len(group)

        # 如果用户数据量大于5条，则随机选择20%的数据归入te,其余归入tr
        if n_items_u >= 5:
            idx = np.zeros(n_items_u, dtype='bool')
            idx[np.random.choice(n_items_u, size=int(test_prop * n_items_u), replace=False).astype('int64')] = True

            tr_list.append(group[np.logical_not(idx)])
            te_list.append(group[idx])

        # 用户数据少于5条 插入tr_list
        else:
            tr_list.append(group)

    data_tr = pd.concat(tr_list)
    data_te = pd.concat(te_list)

    return data_tr, data_te


def numerize(tp, profile2id, show2id):
    uid = tp['userId'].apply(lambda x: profile2id[x])
    sid = tp['movieId'].apply(lambda x: show2id[x])
    return pd.DataFrame(data={'uid': uid, 'sid': sid}, columns=['uid', 'sid'])


if __name__ == '__main__':

    print("Load and Preprocess ML10M dataset")
    # 读取数据
    DATA_DIR = '../dataset/ML10M-ExplicitPositive4Ranking'
    raw_data = pd.read_csv(os.path.join(DATA_DIR, 'ML10M.ExplicitPositive4Ranking.copy1.explicit'), header=None, delimiter=' ')
    vad_raw_data = pd.read_csv(os.path.join(DATA_DIR, 'ML10M.ExplicitPositive4Ranking.copy1.valid'), header=None, delimiter=' ')
    test_raw_data = pd.read_csv(os.path.join(DATA_DIR, 'ML10M.ExplicitPositive4Ranking.copy1.test'), header=None, delimiter=' ')
    raw_data.columns = ['userId', 'movieId', 'rating']
    vad_raw_data.columns = ['userId', 'movieId']
    test_raw_data.columns = ['userId', 'movieId']
    # 保留4分及4分以上的数据
    raw_data = raw_data[raw_data['rating'] > 3.5]

    # 过滤数据
    # 统计每个用户、电影的出现频率
    # 只保留被用户点击了5次以上的物品
    raw_data, user_activity, item_popularity = filter_triplets(raw_data)

    sparsity = 1. * raw_data.shape[0] / (user_activity.shape[0] * item_popularity.shape[0])

    print("After filtering, there are %d watching events from %d users and %d movies (sparsity: %.3f%%)" %
          (raw_data.shape[0], user_activity.shape[0], item_popularity.shape[0], sparsity * 100))

    # 打乱用户顺序
    # unique_uid = user_activity.index
    # np.random.seed(98765)
    # idx_perm = np.random.permutation(unique_uid.size)
    # unique_uid = unique_uid[idx_perm]

    unique_uid = list(range(71567 + 1))

    n_users = len(unique_uid)
    # n_heldout_users = int(n_users * 0.1)

    # # 拆分 Train/Validation/Test 用户
    # tr_users = unique_uid[:(n_users - n_heldout_users * 2)]
    # vd_users = unique_uid[(n_users - n_heldout_users * 2): (n_users - n_heldout_users)]
    # te_users = unique_uid[(n_users - n_heldout_users):]

    train_plays = raw_data
    unique_sid = pd.unique(train_plays['movieId'])

    show2id = dict((sid, i) for (i, sid) in enumerate(unique_sid))
    profile2id = dict((pid, i) for (i, pid) in enumerate(unique_uid))

    pro_dir = os.path.join(DATA_DIR, 'pro_sg')

    if not os.path.exists(pro_dir):
        os.makedirs(pro_dir)

    with open(os.path.join(pro_dir, 'unique_sid.txt'), 'w') as f:
        for sid in unique_sid:
            f.write('%s\n' % sid)

    # 筛选出在训练集中出现过的物品的数据 作为验证集数据
    # ToDo: 不考虑冷启动的问题
    vad_plays = vad_raw_data
    vad_plays = vad_plays.loc[vad_plays['movieId'].isin(unique_sid)]

    # 验证集上进行拆分:
    # 1. 如果用户数据量小于5条，直接归入vad_tr
    # 2. 如果用户数据量大于5条，则随机选择20%的数据归入vad_te,其余归入vad_tr
    vad_plays_tr, vad_plays_te = split_train_test_proportion(vad_plays)

    # 筛选出在训练集中出现过的物品的数据 作为测试集数据
    test_plays = test_raw_data
    test_plays = test_plays.loc[test_plays['movieId'].isin(unique_sid)]

    # 测试集上进行拆分:
    # 1. 如果用户数据量小于5条，直接归入test_tr
    # 2. 如果用户数据量大于5条，则随机选择20%的数据归入test_te,其余归入test_tr
    test_plays_tr, test_plays_te = split_train_test_proportion(test_plays)

    # 所有数据重新编号（根据profile2id和show2id）
    train_data = numerize(train_plays, profile2id, show2id)
    train_data.to_csv(os.path.join(pro_dir, 'train.csv'), index=False)

    vad_data_tr = numerize(vad_plays_tr, profile2id, show2id)
    vad_data_tr.to_csv(os.path.join(pro_dir, 'validation_tr.csv'), index=False)

    vad_data_te = numerize(vad_plays_te, profile2id, show2id)
    vad_data_te.to_csv(os.path.join(pro_dir, 'validation_te.csv'), index=False)

    test_data_tr = numerize(test_plays_tr, profile2id, show2id)
    test_data_tr.to_csv(os.path.join(pro_dir, 'test_tr.csv'), index=False)

    test_data_te = numerize(test_plays_te, profile2id, show2id)
    test_data_te.to_csv(os.path.join(pro_dir, 'test_te.csv'), index=False)

    print("Done!")
