import numpy as np
from sklearn.model_selection import train_test_split


class ItemCF:

    def __init__(self, method):
        """
            train: 已知客户已经对电影的评分
            test: 客户感兴趣的电影但是尚未观看过
        """
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
        ## test
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
        self.similarity(method)

    def similarity(self, method='ItemCF', normalize=True):
        """建立物品相似矩阵"""
        # 1.计算用户相似性 C={C[i][j]}
        self.C = {}  # C[i][j]: 电影i和电影j被同一个用户观看
        N = {}  # N[i]: 用户u买过的产品个数
        for user, items in self.train.items():
            for i in items:
                N[i] = N.get(i, 0) + 1
                if i not in self.C:
                    self.C[i] = {}
                for j in items:
                    if i != j:
                        if method == 'ItemCF':
                            self.C[i][j] = self.C[i].get(j, 0) + 1
                        elif method == 'ItemCF-IUF':
                            self.C[i][j] = self.C[i].get(j, 0) + 1/np.log(1 + len(items))
        # print('C:', self.C)
        # print('N:', N)
        print('计算物品相似度矩阵')

        # 3.对C进行标准化 C_standard={C[i][j]/N[i]/N[j]}
        for i, other_items in self.C.items():
            for j in other_items.keys():
                self.C[i][j] /= np.sqrt(N[i] * N[j])
            if normalize:
                max_Ci = max(self.C[i].values())
                for j in other_items.keys():
                    self.C[i][j] /= max_Ci
        # print('C after standard:', self.C)
        print('标准化相似度矩阵')

    def recommend(self, user, k, N=10):
        """给用户推荐产品"""
        rank = {}
        for i, rui in self.train[user].items():
            for j, wij in sorted(self.C[i].items(), key=lambda x: -x[1])[:k]:
                if j in self.train[user]:
                    continue
                rank[j] = rank.get(j, 0) + wij*1    # 显性用int(rui)，隐性用1
        # print(rank)
        topN = dict(sorted(rank.items(), key=lambda x: -x[1])[:N])
        return topN

    def precision(self, k):
        """精确率 = |tu \cap pu| / |pu|
        tu是测试集对用户u的推荐结果; pu是预测的推荐结果
        """
        fenzi = 0
        fenmu = 0
        for u, item_rate in self.train.items():
            true = self.test.get(u, {})
            pred = self.recommend(u, k)
            fenmu += len(pred)
            for item, rate in pred.items():
                if item in true:
                    fenzi += 1
        print('precision:', fenzi / (fenmu*1.0))
        return fenzi / (fenmu*1.0)

    def recall(self, k):
        """召回率 = |tu \cap pu| / |tu|"""
        fenzi = 0
        fenmu = 0
        for u, item_rate in self.train.items():
            true = self.test.get(u, {})
            pred = self.recommend(u, k)
            fenmu += len(true)
            for item, rate in pred.items():
                if item in true:
                    fenzi += 1
        print('recall:', fenzi / (fenmu * 1.0))
        return fenzi / (fenmu * 1.0)

    def coverage(self, k):
        """覆盖率 = |\cup pu| / |I|
        I是训练集中所有的产品的集合
        """
        pu = set()
        I = set()
        for u, item_rate in self.train.items():
            for i in item_rate.keys():
                I.add(i)
            pred = self.recommend(u, k)
            for i in pred.keys():
                pu.add(i)
        print('coverage:', len(pu) / (len(I) * 1.0))
        return len(pu) / (len(I) * 1.0)

    def popularity(self, k):
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
            pred = self.recommend(u, k)
            for i in pred.keys():
                fenmu += 1
                fenzi += np.log(1 + item_popularity[i])
        print('popularity:', fenzi / (fenmu * 1.0))
        return fenzi / (fenmu * 1.0)


if __name__ == '__main__':
    cf = ItemCF(method='ItemCF-IUF')

    for k in [5, 10, 20, 40, 80, 160]:
        print('-'*10, ' '*3, 'k=', k, ' '*3, "-"*10)
        # 针对推荐系统
        cf.precision(k)
        cf.recall(k)
        # 针对内容提供商
        cf.coverage(k)
        # 针对用户
        cf.popularity(k)
        print('='*30)
    # cf.recommend('A', 3)
