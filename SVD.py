import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


class DataLoader:
    def __init__(self):
        movielen = self.data_load()
        self.train, self.test = train_test_split(movielen, test_size=0.2, random_state=2021, shuffle=True)

    def data_load(self):
        movielen = pd.read_csv('./ml-1m/ratings.dat', sep='::', engine='python', header=None)
        movielen.columns = ['user', 'item', 'rating', 'time']
        return movielen


class SVD:
    """
    prediction:
        \hat r_{ui} = \mu + bi + bu + qi^t*pu
    loss function (has regularization):
        L = \sum_{(u, i) \in R(u,i)} (r_{ui} - \hat r_{ui})^2 + \lambda*(bi^2 + bu^2 + qi^2 + pu^2)
    其中，R(u,i)是用户进行过打分的集合。

    SGD process:
        hyper parameter: learning_rate=\alpha, l2=\lambda
        init bi, bu, qi, pu
        loop epoch:
            loop (u, i, r_{ui}):
                e_{ui} = r_{ui} - \hat r_{ui}
                bi -= \alpha*(-e_{ui}+\lambda*bi)
                bu -= \alpha*(-e_{ui}+\lambda*bu)
                qi -= \alpha*(-e_{ui}*pu + \lambda*qi)
                pu -= \alpha*(-e_{ui}*qi + \lambda*pu)
    """
    def __init__(self, hidden_dim, lr=0.1, l2=0.5, epoch=100):
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.l2 = l2
        self.epoch = epoch

    def make_dictionary(self, data):
        users = data['user'].unique()
        self.user2idx = {user: idx for user, idx in zip(users, range(len(users)))}
        items = data['item'].unique()
        self.item2idx = {item: idx for item, idx in zip(items, range(len(items)))}

    def init(self):
        item_num = len(self.item2idx)
        user_num = len(self.user2idx)
        self.bi = np.random.randn(item_num)
        self.bu = np.random.randn(user_num)
        self.qi = np.random.randn(item_num, self.hidden_dim)
        self.pu = np.random.randn(user_num, self.hidden_dim)

    def train(self, data):
        # 建立user字典和item字典
        self.make_dictionary(data)
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
                self.bi[item_idx] -= self.lr*(-eui + self.l2*self.bi[item_idx])
                self.bu[user_idx] -= self.lr*(-eui + self.l2*self.bu[user_idx])
                self.qi[item_idx] -= self.lr*(-eui*self.pu[user_idx] + self.l2*self.qi[item_idx])
                self.pu[user_idx] -= self.lr*(-eui*self.qi[item_idx] + self.l2*self.pu[user_idx])

                ## 增加rmse
                rmse += eui**2
            self.lr *= 0.95
            print('epoch %s | training error=%.4f' % (epoch+1, np.sqrt(rmse/len(data))))

    def predict(self, user, item):
        user_idx = self.user2idx[user]
        item_idx = self.item2idx[item]
        rui_hat = self.mu + self.bi[item_idx] + self.bu[user_idx] + self.qi[item_idx].dot(self.pu[user_idx].T)
        if rui_hat > 5:
            rui_hat = 5
        if rui_hat < 1:
            rui_hat = 1
        return rui_hat

    def eval(self, test):
        rmse = 0
        for user, item, rui in zip(test['user'], test['item'], test['rating']):
            if user in self.user2idx and item in self.item2idx:
                rui_hat = self.predict(user, item)
                eui = rui - rui_hat
                rmse += eui**2
            elif user not in self.user2idx:
                print('user %s does not exist' % user)
            elif item not in self.item2idx:
                print('item %s does not exist' % item)
        print('rmse of test set is %.4f' % np.sqrt(rmse/len(test)))


if __name__ == '__main__':
    loader = DataLoader()
    print('train set {}'.format(loader.train.shape))
    print('test set {}'.format(loader.test.shape))
    recommender = SVD(hidden_dim=20, epoch=20, lr=0.04, l2=0.2)
    recommender.train(loader.train)
    recommender.eval(loader.test)
