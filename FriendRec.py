import random
from math import sqrt

class FriendRecommend:

    def __init__(self):

        with open('./Slashdot/Slashdot0902.txt') as f:
            # 忽略前四行
            nodes = set()
            side_num = 0
            for i in range(4):
                f.readline()
            data = []
            while True:
                line = f.readline()
                if len(line) == 0:
                    break
                fromnode, tonode = line.strip().split('\t')
                data.append([fromnode, tonode])
                nodes.add(fromnode)
                nodes.add(tonode)
                side_num += 1
        # data = [[1,2],[2,1],[1,3],[3,2],[1,4],[4,2]]
        print('nodes num:', len(nodes))
        print('line num:', side_num)
        print('train test split')
        self.train_test_split(data)
        print('create in dictionary and out dictionary')
        self.initgraph()

    def reshape_test(self, method):
        test = {}
        if method in ['out', 'out-in']:
            for fromnode, innode in self.test:
                if fromnode not in test:
                    test[fromnode] = []
                test[fromnode].append(innode)
        elif method == 'in':
            for fromnode, innode in self.test:
                if innode not in test:
                    test[innode] = []
                test[innode].append(fromnode)
        return test

    def train_test_split(self, data):
        self.test = []
        self.train = []
        for d in data:
            if random.random()<0.1:
                self.test.append(d)
            else:
                self.train.append(d)
        # print('train:', self.train)
        # print('test:', self.test)

    def initgraph(self):
        self._out = {}
        self._in = {}
        for fromnode, innode in self.train:
            if fromnode not in self._out:
                self._out[fromnode] = []
            self._out[fromnode].append(innode)
            if innode not in self._in:
                self._in[innode] = []
            self._in[innode].append(fromnode)
        # print('out:', self._out)
        # print('in:', self._in)

    def get_friends(self, user):
        if user in self._out:
            return self._out[user]
        else:
            return []

    def recommend(self, user, method='out', N=10):
        topN = {}
        if method=='out':
            # 把user关注的人视为好友 a->b<-c, 将c推荐给a
            rank = {}
            for fromid in self._out[user]:
                if fromid is self._in:
                    for inid in self._in[fromid]:
                        friends = self.get_friends(user)
                        if inid in friends:
                            continue
                        rank[inid] = rank.get(inid, 0) + 1
            rank_norm = {inid: share/sqrt(len(self._out[inid])*len(self._out[user])) for inid, share in rank.items()}
            topN = dict(sorted(rank_norm.items(), key=lambda x: -x[1])[:N])
        elif method=='in':
            # 有着共同的粉丝 a<-b->c, 将c推荐给a
            rank = {}
            for fans in self._in[user]:
                if fans in self._out:
                    for star in self._out[fans]:
                        friends = self.get_friends(user)
                        if star in friends:
                            continue
                        rank[star] = rank.get(star, 0) + 1
            rank_norm = {star: share/sqrt(len(self._in[user])*len(self._in[star])) for star, share in rank.items()}
            topN = dict(sorted(rank_norm.items(), key=lambda x: -x[1])[:N])
        elif method=='out-in':
            # a->b->c 将c推荐给a
            rank = {}
            for star in self._out[user]:
                if star in self._out:
                    for star2 in self._out[star]:
                        friends = self.get_friends(user)
                        if star2 in friends:
                            continue
                        rank[star2] = rank.get(star2, 0) + 1
            rank_norm = {star2: share/sqrt(len(self._out[user])*len(self._in[star2])) for star2, share in rank.items()}
            topN = dict(sorted(rank_norm.items(), key=lambda x: -x[1])[:N])
        return topN

    def precision(self, method):
        """准确率"""
        test = self.reshape_test(method)
        fenzi = 0
        fenmu = 0
        users = self._out.keys() if method in ['out', 'out-in'] else self._in.keys()
        for user in users:
            true = test.get(user, {})
            pred = self.recommend(user, method=method)
            # print(user, pred)
            fenmu += len(pred)
            for f in pred.keys():
                if f in true:
                    fenzi += 1
        print('precision: ', fenzi / (fenmu * 1.0))
        return fenzi / (fenmu * 1.0)

    def recall(self, method):
        test = self.reshape_test(method)
        # calculate
        fenzi = 0
        fenmu = 0
        users = self._out.keys() if method in ['out', 'out-in'] else self._in.keys()
        for user in users:
            true = test.get(user, [])
            pred = self.recommend(user, method=method)
            fenmu += len(true)
            for f in pred.keys():
                if f in true:
                    fenzi += 1
        print('recall: ', fenzi / (fenmu * 1.0))
        return fenzi / (fenmu * 1.0)


if __name__ == '__main__':
    fr = FriendRecommend()
    # for method in ['out-in', 'in', 'out']:
    #     print('-' * 10, ' ' * 3, 'method=', method, ' ' * 3, "-" * 10)
    #     fr.precision(method=method)
    #     fr.recall(method=method)
    fr.precision(method='in')
    fr.recall(method='in')
