import numpy as np
from sklearn.model_selection import train_test_split


class UserCF:

    def __init__(self):
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
        self.similarity()

    def similarity(self, method='UserCF'):
        """建立用户相似矩阵"""
        # 1.建立item_user倒排表
        item_user = {}
        for user, movie_rate in self.train.items():
            for movie, rate in movie_rate.items():
                if movie not in item_user:
                    item_user[movie] = set()
                item_user[movie].add(user)
        print('建立产品倒排表')

        # 2.计算用户相似性 C={C[u][v]}
        self.C = {}  # C[u][v]: 用户u和用户v共同购买过的产品数
        N = {}  # N[u]: 用户u买过的产品个数
        for item, users in item_user.items():
            for u in users:
                N[u] = N.get(u, 0) + 1
                if u not in self.C:
                    self.C[u] = {}
                for v in users:
                    if v != u:
                        if method == 'UserCF':
                            self.C[u][v] = self.C[u].get(v, 0) + 1
                        elif method == 'User-IIF':      # 对热门商品进行惩罚
                            self.C[u][v] = self.C[u].get(v, 0) + 1/np.log(1 + len(users))
        # print('C:', self.C)
        # print('N:', N)
        print('计算用户相似度矩阵')

        # 3.对C进行标准化 C_standard={C[u][v]/N[u]/N[v]}
        for u, other_users in self.C.items():
            for v in other_users.keys():
                self.C[u][v] /= np.sqrt(N[u] * N[v])
        # print('C after standard:', self.C)
        print('标准化相似度矩阵')

    def recommend(self, user, k, N=10):
        """给用户推荐产品"""
        rank = {}
        similar_users = sorted(self.C[user].items(), key=lambda x: -x[1])[:k]   # 选取top k位与u相似的用户
        for v, wuv in similar_users:
            for i, rvi in self.train[v].items():
                if i in self.train[user]:
                    continue
                rank[i] = rank.get(i, 0) + wuv*int(1)     # 按照topn的原理，rvi=1
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
    cf = UserCF()

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
