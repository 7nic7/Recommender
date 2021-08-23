import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt
tf.compat.v1.disable_eager_execution()


class DataLoader:
    """
    之所以要将数据处理为x_value和x_index的形式，是因为稀疏矩阵one-hot编码的储存大，且查询耗时
    如 有三个变量，前两个col1, col2 均为连续变量，第三个变量是离散变量col3=[0,1,2]
    - 那么常规做法是：
        v 的维度是 5*k (2+3 = 5, k表示隐向量的长度)
        x 的维度是 n*5 (n表示数据量)
    在计算Y时，sum(multiply(x, v), axis=1) 得到n*k的矩阵
    - 但是引入了x_index后
        v 的维度是 5*k, 不变
        x_value 的维度是 n*3
        x_index 的维度是 n*3
    在计算Y时，1.m = embedding_lookup(v, x_index) 得到维度为n*3*k的矩阵  这一步加快了运行速度
             2.sum(multiply(x_value, m), axis=1) 得到n*k的矩阵
    当然上述只是举个例子，在实际应用中，可能离散变量的值更多，而不是只有3个，从而加速效果更好。
    """
    def __init__(self):
        # 数据读取
        self.load_data()

    def load_data(self):
        # 读取user特征数据
        self.users = pd.read_csv('./ml-1m/users.dat', sep='::', index_col=0,
                                 names=['sex', 'age', 'job', 'zipcode'], engine='python')
        # 读取评分数据
        self.ratings = pd.read_csv('./ml-1m/ratings.dat', sep='::',
                                   names=['uid', 'iid', 'rate', 'timestamp'], engine='python')
        self.ratings.drop('timestamp', axis=1, inplace=True)
        self.target_name = 'rate'

    def process(self):
        # 处理标签
        ## 暂时将5分定义为1，其他为0；当然也可以如LFM中那样，考虑负采样
        self.process_target()
        # 将user特征和rating的user_index以及item_index合并
        self.users = self.users.reset_index()
        data = self.ratings.merge(self.users, how='left', left_on='uid', right_on='index')
        data.drop('index', axis=1, inplace=True)
        # 生成每一列中value转index的字典
        feat_dict = self.gen_dict(data, continuous_vars=[])
        # train_test_split
        train, test = train_test_split(data, test_size=0.2, shuffle=True, random_state=2021)
        # 创建x_val 和x_index 数据
        train_value, train_index = self.value_and_index(train, feat_dict, continuous_vars=[])
        test_value, test_index = self.value_and_index(test, feat_dict, continuous_vars=[])
        self.feat_dim_v1 = len([col for col in train_value.columns if col !=self.target_name])
        # print('train_value:\n', train_value.head())
        # print('train_index:\n', train_index.head())
        # print('test_value:\n', test_value.head())
        # print('test_index:\n', test_index.head())
        return train_value, train_index, test_value, test_index

    def gen_dict(self, data, continuous_vars=None):
        feat_dict = {}
        cnt = 0
        for col in data.columns:
            if col != self.target_name:
                if col in continuous_vars:  # 连续变量
                    feat_dict[col] = cnt
                    cnt += 1
                else:   # 离散变量
                    distinct_vals = data[col].unique()
                    feat_dict[col] = {k: v for k, v in zip(distinct_vals, range(cnt, cnt+len(distinct_vals)))}
                    cnt += len(distinct_vals)
        self.feat_dim_v2 = cnt
        return feat_dict

    def value_and_index(self, data, feat_dict, continuous_vars=None):
        x_value = data.copy()
        x_index = data.copy()
        for col in data.columns:
            if col != self.target_name:
                if col in continuous_vars:  # 连续变量
                    x_index[col] = feat_dict[col]
                else:   # 离散变量
                    x_index[col] = x_index[col].map(feat_dict[col])
                    x_value[col] = 1.
        return x_value, x_index

    def process_target(self):
        self.ratings['rate'] = self.ratings['rate'].map({1: 0, 2: 0, 3: 0, 4: 0, 5: 1})


class FM:
    """
    计算CTR
    因子分解机 —— Factorization Machines
    Y = w_0 + \sum w_k*x_k + 1/2*\sum_f [(\sum_i v_{i,f}*x_i)^2 - \sum_i (v_{i,f}*x_i)^2]
    SGD: tensorflow自动算


    与LFM相比的优势是：
        可以考虑除了item向量和user向量以外的特征，如user的年龄、item的类别等
    """

    def __init__(self, embed_dim, feat_dim_v1, feat_dim_v2, epoch, batch_size):
        self.embed_dim = embed_dim      # k
        self.feat_dim_v1 = feat_dim_v1      # x_value的列维度-m1
        self.feat_dim_v2 = feat_dim_v2      # one-hot后的列维度-m2
        self.epoch = epoch
        self.batch_size = batch_size

        self.weights = {}

    def build(self):
        self.x_value = tf.compat.v1.placeholder(shape=[None, self.feat_dim_v1], dtype=tf.float32)
        self.x_index = tf.compat.v1.placeholder(shape=[None, self.feat_dim_v1], dtype=tf.int32)
        self.y = tf.compat.v1.placeholder(shape=[None], dtype=tf.float32)

        with tf.name_scope('embedding_lookup'):
            self.weights['w'] = tf.compat.v1.Variable(tf.random.normal([self.feat_dim_v2, 1], stddev=0.05), name='w')
            self.weights['v'] = tf.compat.v1.Variable(tf.random.normal([self.feat_dim_v2, self.embed_dim], stddev=0.05), name='v')
            w_embed = tf.compat.v1.nn.embedding_lookup(self.weights['w'], self.x_index)     # batch*m1*1
            v_embed = tf.compat.v1.nn.embedding_lookup(self.weights['v'], self.x_index)     # batch*m1*k

        with tf.name_scope('one_factor'):
            w_x = tf.multiply(tf.reshape(w_embed, [-1, self.feat_dim_v1]), self.x_value)
            part1 = tf.reduce_sum(w_x, axis=1)  # batch*1

        with tf.name_scope('mutual_factor'):
            v_x = tf.multiply(v_embed, tf.reshape(self.x_value, [-1, self.feat_dim_v1, 1]))
            # 先求和再平方
            part2_1 = tf.square(tf.reduce_sum(v_x, axis=1))     # batch*k
            # 先平方再求和
            part2_2 = tf.reduce_sum(tf.square(v_x), axis=1)     # batch*k
            # 得到交互项的最终结果
            part2 = 1/2 * tf.reduce_sum(part2_1 - part2_2, axis=1)

        with tf.name_scope('optimizer'):
            b = tf.Variable([0.], name='w0')
            logits = b + part1 + part2
            self.prediction = tf.compat.v1.sigmoid(logits)
            pred_label = tf.cast(tf.argmax(tf.concat([tf.reshape(1-self.prediction, [-1,1]), tf.reshape(self.prediction, [-1,1])], axis=1), axis=1), dtype=tf.float32)
            self.accuary = tf.reduce_mean(tf.cast(tf.equal(self.y, pred_label), dtype=tf.float32))
            self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.y, logits))
            # self.loss = -tf.reduce_mean(self.y*tf.compat.v1.log(tf.clip_by_value(self.prediction, 1e-10, 1.))+(1-self.y)*tf.compat.v1.log(tf.clip_by_value(1-self.prediction, 1e-10, 1.)), axis=0)
            self.op = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-4).minimize(self.loss)

    def get_batch_data(self, x_value, x_index, y):
        n = x_value.shape[0]

        for epo in range(self.epoch):
            print('---------------------epoch %s---------------------'%epo)
            # shuffle
            index = list(range(n))
            np.random.shuffle(index)
            np.random.shuffle(index)
            np.random.shuffle(index)
            x_value = x_value[index]
            x_index = x_index[index]
            y = y[index]
            start_index = 0
            for batch in range(n//self.batch_size):
                x_value_batch = x_value[start_index:(start_index+self.batch_size)]
                x_index_batch = x_index[start_index:(start_index+self.batch_size)]
                y_batch = y[start_index:(start_index+self.batch_size)]
                start_index += self.batch_size
                yield x_value_batch, x_index_batch, y_batch

    def train_model(self, x_value, x_index, y):
        train_loss_list = []
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            for i, (x_value_batch, x_index_batch, y_batch) in enumerate(self.get_batch_data(x_value, x_index, y)):
                _, train_loss, train_acc = sess.run([self.op, self.loss, self.accuary],
                                                    feed_dict={self.x_value: x_value_batch,
                                                               self.x_index: x_index_batch,
                                                               self.y: y_batch})
                if (i+1) % 100 == 0:
                    print('batch %s: train loss=%.5f, train_acc=%.3f' % (i+1, train_loss, train_acc))
                    train_loss_list.append(train_loss)
            plt.plot(train_loss_list)
            plt.show()

    def evaluate(self):
        pass


if __name__ == '__main__':
    loader = DataLoader()
    train_value, train_index, test_value, test_index = loader.process()
    print('feat_dim_v1', loader.feat_dim_v1)
    print('feat_dim_v2', loader.feat_dim_v2)
    model = FM(5, loader.feat_dim_v1, loader.feat_dim_v2, 20, 1024)
    model.build()
    x_value = train_value.drop('rate', axis=1).values
    x_index = train_index.drop('rate', axis=1).values
    y = train_value['rate'].values
    print(type(x_value))
    print(x_value.shape)
    model.train_model(x_value, x_index, y)
