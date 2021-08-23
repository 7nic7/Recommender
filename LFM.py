import numpy as np
import pandas as pd
from math import exp
from sklearn.model_selection import train_test_split


class LFM:
    """
    Latent Factor Model
    """

    def __init__(self, F, alpha, _lambda, ratio, save=True):
        """
        :param F: 隐特征的个数
        :param alpha: 学习速率
        :param _lambda: 正则化参数
        :param ratio: 负样本/正样本比例
        """
        self.F = F
        self.alpha = alpha
        self._lambda = _lambda
        self.ratio = ratio

        # 开始训练更新参数p和q
        self.build()
        self.train_model()

        # 保存p和q
        if save:
            self.save()

    def _init_data(self):
        """读取数据"""
        # load data
        f = open('./ml-1m/ratings.dat')
        lines = []
        for line in f:
            user, movie, rate, timestamp = line.split('::')
            lines.append([user, movie, rate])

        # train test split
        self.train, self.test = train_test_split(lines, test_size=0.125, random_state=2021)
        print('train length:', len(self.train))
        print('test length:', len(self.test))

        # list -> dict
        train = {}
        for user, movie, rate in self.train:
            if user not in train:
                train[user] = {}
            train[user][movie] = rate
        self.train = train

        test = {}
        for user, movie, rate in self.test:
            if user not in test:
                test[user] = {}
            test[user][movie] = rate
        self.test = test

        # test
        # self.train = {
        #     'A': {'a':1, 'b':1},
        #     'B': {'a':1, 'c':1},
        #     'C': {'b':1, 'e':1},
        #     'D': {'c':1, 'd':1}
        # }
        # self.test= {
        #     'A': {'d':1},
        #     'D': {'e':1}
        # }

    def _init_model(self):
        """初始化模型——主要是初始化p和q向量"""
        # 统计训练集中items和users集合
        users = set()
        self.items = set()
        for u, i in self.train.items():
            users.add(u)
            self.items.update(set(i))

        # 初始化向量
        self.p = pd.DataFrame(np.random.randn(len(users), self.F), index=users)    # 用户向量
        self.q = pd.DataFrame(np.random.randn(len(self.items), self.F), index=self.items)    # 物品向量
        # print('users set:', self.p)
        # print('items set:', self.q)

    def _optimize(self, user, item, rui):
        """
        loss function = \sum_{u,i} (rui - pu*qi)^2 + \lambda ||pu||^2 + \lambda ||qi||^2
        p_grad = -2* (rui - pu*qi)*qi + 2*\lambda pu
        q_grad = -2* (rui - pu*qi)*pu + 2*\lambda qi
        记 error = (rui - pu*qi)

        optimize:
            p = p - \alpha* p_grad
            q = q - \alpha* q_grad
        """
        # 计算梯度
        pred = self.p.loc[user].to_numpy().dot(self.q.loc[item].to_numpy())
        pred_prob = 1./(1 + exp(-pred)) # sigmoid 防止梯度爆炸
        error = rui - pred_prob
        p_grad = -error*self.q.loc[item].to_numpy() + self._lambda*self.p.loc[user].to_numpy()
        q_grad = -error*self.p.loc[user].to_numpy() + self._lambda*self.q.loc[item].to_numpy()
        # optimize
        self.p.loc[user] = self.p.loc[user].to_numpy() - self.alpha* p_grad
        self.q.loc[item] = self.q.loc[item].to_numpy() - self.alpha* q_grad

    def _neg_sample(self, user_items):
        for user, items in user_items.items():
            # 正样本
            pos_samples = set(items)
            # 负样本
            neg_num = int(self.ratio*len(pos_samples))
            try:
                neg_samples = np.random.choice(list(self.items - pos_samples), neg_num, replace=False)
            except ValueError:
                neg_samples = np.random.choice(list(self.items - pos_samples), neg_num, replace=True)

            samples = pos_samples.union(set(neg_samples))
            for item in samples:
                rui = 1 if item in pos_samples else 0
                yield user, item, rui

    def build(self):
        self._init_data()
        self._init_model()

    def train_model(self, max_step=30):
        delta = np.inf
        step = 0
        old_p, old_q = self.p.copy(), self.q.copy()
        while delta > 1e-8 and step<max_step:
            print('step: ', step, '| delta = ', delta)
            for user, item, rui in self._neg_sample(self.train):
                self._optimize(user, item, rui)
            self.alpha *= 0.9
            step += 1
            delta = np.sum((old_p.to_numpy()-self.p.to_numpy())**2) + np.sum((old_q.to_numpy()-self.q.to_numpy())**2)
            old_p, old_q = self.p.copy(), self.q.copy()

    def save(self):
        self.p.to_csv('./output/p(F=%s,alpha=%s,lambda=%s,ratio=%s).csv' % (self.F, self.alpha, self._lambda, self.ratio))
        self.q.to_csv('./output/q(F=%s,alpha=%s,lambda=%s,ratio=%s).csv' % (self.F, self.alpha, self._lambda, self.ratio))

    def recommend(self, user, N=10):
        rank = {}
        for item in self.q.index:
            if item in self.train[user]:
                continue
            rank[item] = self.p.loc[user].to_numpy().dot(self.q.loc[item].to_numpy())
        topN = dict(sorted(rank.items(), key=lambda x: -x[1])[:N])
        # print(topN)
        return topN

    def precision(self):
        """精确率 = |tu \cap pu| / |pu|
        tu是测试集对用户u的推荐结果; pu是预测的推荐结果
        """
        fenzi = 0
        fenmu = 0
        for u, item_rate in self.train.items():
            true = self.test.get(u, {})
            pred = self.recommend(u)
            fenmu += len(pred)
            for item, rate in pred.items():
                if item in true:
                    fenzi += 1
        print('precision:', fenzi / (fenmu*1.0))
        return fenzi / (fenmu*1.0)

    def recall(self):
        """召回率 = |tu \cap pu| / |tu|"""
        fenzi = 0
        fenmu = 0
        for u, item_rate in self.train.items():
            true = self.test.get(u, {})
            pred = self.recommend(u)
            fenmu += len(true)
            for item, rate in pred.items():
                if item in true:
                    fenzi += 1
        print('recall:', fenzi / (fenmu * 1.0))
        return fenzi / (fenmu * 1.0)

    def coverage(self):
        """覆盖率 = |\cup pu| / |I|
        I是训练集中所有的产品的集合
        """
        pu = set()
        I = set()
        for u, item_rate in self.train.items():
            for i in item_rate.keys():
                I.add(i)
            pred = self.recommend(u)
            for i in pred.keys():
                pu.add(i)
        print('coverage:', len(pu) / (len(I) * 1.0))
        return len(pu) / (len(I) * 1.0)

    def popularity(self):
        """度量推荐的新颖性，用平均流行度来衡量"""
        # 1.计算各个电影的流行度，即计算电影被观看的人次
        item_popularity = {}
        for u, item_rate in self.train.items():
            for item in item_rate.keys():
                item_popularity[item] = item_popularity.get(item, 0) + 1

        # 计算预测出的电影推荐列表的平均流行度
        fenzi = 0
        fenmu = 0
        for u in self.train.keys():
            pred = self.recommend(u)
            for i in pred.keys():
                fenmu += 1
                fenzi += np.log(1 + item_popularity[i])
        print('popularity:', fenzi / (fenmu * 1.0))
        return fenzi / (fenmu * 1.0)


if __name__ == '__main__':
    for ratio in [1, 2, 3, 5, 10, 20]:
        print('-' * 10, ' ' * 3, 'ratio=', ratio, ' ' * 3, "-" * 10)
        lfm = LFM(F=100, alpha=0.02, _lambda=0.01, ratio=ratio)

        lfm.precision()
        lfm.recall()
        lfm.coverage()
        lfm.popularity()
