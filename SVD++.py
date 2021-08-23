from recommendation.SVD import *


class SVDPlusPlus(SVD):
    """
    SVD++ & SVD:
        SVD++与SVD相比，增加了对用户隐性喜欢的度量，这样可以将用户的评分记录和历史浏览记录一起进行建模。
    而SVD只能利用用户的评分记录。

    prediction:
        \hat r_{ui} = \mu + bi + bu + qi^t*(pu + 1/\sqrt|I(u)|*\sum_{j \in I(u)} yj)
    其中，I(u)表示用户u进行浏览或打分的item集合。

    loss function:
        L = \sum_{(u, i) \in R(u,i)} (r_{ui} - \hat r_{ui})^2 + \lambda*(bi^2 + bu^2 + qi^2 + pu^2 + yj^2)

    SGD process:
        hyper parameter: learning_rate=\alpha, l2=\lambda
        init bi, bu, qi, pu
        loop epoch:
            loop (u, i, r_{ui}):
                e_{ui} = r_{ui} - \hat r_{ui}
                bi -= \alpha*(-e_{ui}+\lambda*bi)
                bu -= \alpha*(-e_{ui}+\lambda*bu)
                qi -= \alpha*(-e_{ui}*(pu + 1/\sqrt|I(u)|*\sum_{j \in I(u)} yj) + \lambda*qi)
                pu -= \alpha*(-e_{ui}*qi + \lambda*pu)
                loop j:
                    yj -= \alpha*(-e_{ui}*1/\sqrt|I(u)|*qi + \lambda*yj)
    """
    def __init__(self, hidden_dim, lr=0.1, l2=0.04, epoch=100):
        super(SVDPlusPlus, self).__init__(hidden_dim, lr, l2, epoch)

    def item_stat(self, data):
        self.user2item = {}
        for user, item in zip(data['user'], data['item']):
            if user not in self.user2item:
                self.user2item[user] = []
            self.user2item[user].append(item)

    def init(self):
        super(SVDPlusPlus, self).init()
        item_num = len(self.bi)
        self.yj = np.random.randn(item_num, self.hidden_dim)

    def train(self, data):
        # 建立user字典和item字典
        self.make_dictionary(data)
        # 统计下user的历史浏览或购买的商品集合
        self.item_stat(data)
        # 初始化参数
        self.mu = data['rating'].mean()
        self.init()
        # 开始训练
        for epoch in range(self.epoch):
            data.sample(frac=1).reset_index(drop=True)
            rmse = 0
            for user, item, rui in zip(data['user'], data['item'], data['rating']):
                user_idx = self.user2idx[user]
                item_idx = self.item2idx[item]

                ## 计算error
                rui_hat = self.predict(user, item)
                eui = rui - rui_hat
                ## SGD
                item_num = len(self.user2item[user])
                item_sum = sum(map(lambda x: self.yj[self.item2idx[x]], self.user2item[user]))
                self.bi[item_idx] -= self.lr * (-eui + self.l2 * self.bi[item_idx])
                self.bu[user_idx] -= self.lr * (-eui + self.l2 * self.bu[user_idx])
                self.qi[item_idx] -= self.lr * (-eui * (self.pu[user_idx] + 1/np.sqrt(item_num)*item_sum) + self.l2 * self.qi[item_idx])
                self.pu[user_idx] -= self.lr * (-eui * self.qi[item_idx] + self.l2 * self.pu[user_idx])

                for item_j in self.user2item[user]:
                    item_j_idx = self.item2idx[item_j]
                    self.yj[item_j_idx] -= self.lr * (-eui * 1/np.sqrt(item_num) * self.qi[item_idx] + self.l2 * self.yj[item_j_idx])

                ## 增加rmse
                rmse += eui ** 2
            self.lr *= 0.95
            print('epoch %s | training error=%.4f' % (epoch + 1, np.sqrt(rmse / len(data))))

    def predict(self, user, item):
        user_idx = self.user2idx[user]
        item_idx = self.item2idx[item]
        item_num = len(self.user2item[user])
        item_sum = sum(map(lambda x: self.yj[self.item2idx[x]], self.user2item[user]))
        rui_hat = self.mu + self.bi[item_idx] + self.bu[user_idx] + \
                  self.qi[item_idx].dot(self.pu[user_idx].T + 1/np.sqrt(item_num)*item_sum)
        if rui_hat > 5:
            rui_hat = 5
        if rui_hat < 1:
            rui_hat = 1
        return rui_hat


if __name__ == '__main__':
    loader = DataLoader()
    print('train set {}'.format(loader.train.shape))
    print('test set {}'.format(loader.test.shape))
    recommender = SVDPlusPlus(hidden_dim=20, epoch=20, lr=0.04, l2=0.2)
    recommender.train(loader.train)
    recommender.eval(loader.test)
