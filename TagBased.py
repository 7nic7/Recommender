import pandas as pd
import numpy as np
import random


class TagBasedRecommend:

    def __init__(self):
        self.data = pd.read_csv('./delicious/user_taggedbookmarks-timestamps.dat', sep='\t')
        # self.data = pd.DataFrame({
        #     'userID': [8,8,8,8,8,9,9,9,1,1],
        #     'bookmarkID': [1,2,7,7,7,1,2,7,7,7],
        #     'tagID': [1,1,1,6,7,1,1,1,6,7]
        # })
        # print(self.data.head())
        print('train test split')
        self.train_test_split()
        print('initialize the dictionaries')
        self.initstat()

    def train_test_split(self, test_size=0.1):
        """以user和item为分割依据，即同一个user看的书所打标签要么在训练集中要么在测试集中"""

        def add_data(data, todict, key1, key2):
            if key1 not in todict:
                todict[key1] = {}
            if key2 not in todict[key1]:
                todict[key1][key2] = []
            todict[key1][key2].append(data)

        # 1.建立user_item_tag字典
        # {user1: {item1: [tag1, tag2], item2: [tag1]}}
        user_item_tag = {}
        for _, row in self.data.iterrows():
            user, item, tag = row['userID'], row['bookmarkID'], row['tagID']
            add_data(tag, user_item_tag, user, item)

        # 2.train test split
        self.train, self.test = {}, {}
        for u in user_item_tag.keys():
            for i in user_item_tag[u].keys():
                if random.random()<test_size:  # 给测试集
                    for t in user_item_tag[u][i]:
                        add_data(t, self.test, u, i)
                else:
                    for t in user_item_tag[u][i]:
                        add_data(t, self.train, u, i)
        # print('train:', self.train)
        # print('test:', self.test)

    def initstat(self):
        """
        user的标签集合 user_tag={user:{tag:n}}
        tag的商品集合 tag_item={tag:{item:m}}
        user的item集合 user_item={user:{item:n}}
        """
        def addone(data, todict, key1, key2):
            if key1 not in todict:
                todict[key1] = {}
            todict[key1][key2] = todict[key1].get(key2, 0) + data

        self.user_tag = {}
        self.user_item = {}
        self.tag_item = {}

        self.item_tag = {}
        self.tag_user = {}
        self.item_user = {}
        for u, items in self.train.items():
            for i, tags in items.items():
                for t in tags:
                    addone(1, self.user_tag, u, t)
                    addone(1, self.user_item, u, i)
                    addone(1, self.tag_item, t, i)

                    addone(1, self.item_tag, i, t)
                    addone(1, self.tag_user, t, u)
                    addone(1, self.item_user, i, u)
        # print('user_tag:', self.user_tag)
        # print('user_item:', self.user_item)
        # print('tag_item:', self.tag_item)
        # print('item_tag:', self.item_tag)
        # print('tag_user:', self.tag_user)
        # print('item_user:', self.item_user)

    def recommend(self, user, method='TFIDF', N=10):
        """给用户推荐商品，通过标签"""
        rank = {}
        for t, n_ut in self.user_tag[user].items():
            for i, n_ti in self.tag_item[t].items():
                if i in self.user_item[user]:
                    continue
                if method=='SIMPLE':
                    rank[i] = rank.get(i, 0) + n_ut*n_ti
                elif method=='TFIDF':   # 对热门的标签进行惩罚,分母是标签被多少个不同的用户使用过
                    n_t = len(self.tag_user[t])
                    rank[i] = rank.get(i, 0) + n_ut*n_ti/np.log(1+n_t)
                elif method=='TFIDF++': # 对热门标签和热门商品均进行惩罚
                    n_t = len(self.tag_user[t])
                    n_i = len(self.item_user[i])
                    rank[i] = rank.get(i, 0) + n_ut*n_ti/np.log(1+n_t)/np.log(1+n_i)
        # print('rank:', rank)
        topN = dict(sorted(rank.items(), key=lambda x: -x[1])[:N])
        return topN

    def precision(self, method):
        """精确率 = |tu \cap pu| / |pu|
        tu是测试集对用户u的推荐结果; pu是预测的推荐结果
        """
        fenzi = 0
        fenmu = 0
        for u in self.train.keys():
            true = self.test.get(u, {})
            pred = self.recommend(u, method)
            fenmu += len(pred)
            for item in pred.keys():
                if item in true:
                    fenzi += 1
        print('precision:', 0 if fenmu==0 else fenzi / (fenmu * 1.0))
        return 0 if fenmu==0 else fenzi / (fenmu * 1.0)

    def recall(self, method):
        """召回率 = |tu \cap pu| / |tu|"""
        fenzi = 0
        fenmu = 0
        for u in self.train.keys():
            true = self.test.get(u, {})
            pred = self.recommend(u, method)
            fenmu += len(true)
            for item in pred.keys():
                if item in true:
                    fenzi += 1
        print('recall:', 0 if fenmu==0 else fenzi / (fenmu * 1.0))
        return 0 if fenmu==0 else fenzi / (fenmu * 1.0)

    def coverage(self, method):
        """覆盖率 = |\cup pu| / |I|
        I是训练集中所有的产品的集合
        """
        pu = set()
        I = set()
        for u, item_tag in self.train.items():
            for i in item_tag.keys():
                I.add(i)
            pred = self.recommend(u, method)
            for i in pred.keys():
                pu.add(i)
        print('coverage:', len(pu) / (len(I) * 1.0))
        return len(pu) / (len(I) * 1.0)

    def popularity(self, method):
        """度量推荐的新颖性，用平均流行度来衡量"""
        # 1.计算各个电影的流行度，即计算电影被观看的人次
        item_popularity = {}
        for u, item_tag in self.train.items():
            for item in item_tag.keys():
                item_popularity[item] = item_popularity.get(item, 0) + 1

        # 计算预测出的电影推荐列表的平均流行度
        fenzi = 0
        fenmu = 0
        for u in self.train.keys():
            pred = self.recommend(u, method)
            for i in pred.keys():
                fenmu += 1
                fenzi += np.log(1 + item_popularity[i])
        print('popularity:', 0 if fenmu==0 else fenzi / (fenmu * 1.0))
        return 0 if fenmu==0 else fenzi / (fenmu * 1.0)

    def diversity(self, method):
        """多样性，是相似度的相反数
        1- \sum_u \sum_(i,j) cos(i,j) / \sum_u \sum_(i,j) 1
        """
        def cosine(i, j):
            # 计算item i 与item j之间的相关性, 是通过标签来评估商品之前的相似性
            fenzi = 0
            for t, w_i in self.item_tag[i].items():
                if t in self.item_tag[j]:
                    fenzi += w_i * self.item_tag[j][t]

            fenmu_mul1 = sum([w_i**2 for w_i in self.item_tag[i].values()])
            fenmu_mul2 = sum([w_j**2 for w_j in self.item_tag[j].values()])
            return fenzi / np.sqrt(fenmu_mul1 * fenmu_mul2)

        fenzi = 0
        fenmu = 0
        for u in self.train.keys():
            pred = self.recommend(u, method)
            for i in pred.keys():
                for j in pred.keys():
                    if i == j:
                        continue
                    fenzi += cosine(i, j)
                    fenmu += 1
        print('diversity:', 0 if fenmu==0 else 1-fenzi / (fenmu * 1.0))
        return 0 if fenmu==0 else 1-fenzi / (fenmu * 1.0)


if __name__ == '__main__':
    tagbased = TagBasedRecommend()
    for method in ['SIMPLE', 'TFIDF', 'TFIDF++']:
        print('-' * 10, ' ' * 3, 'method=', method, ' ' * 3, "-" * 10)
        tagbased.precision(method)
        tagbased.recall(method)
        tagbased.coverage(method)
        tagbased.popularity(method)
        tagbased.diversity(method)
