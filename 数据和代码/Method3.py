import numpy as np
import pandas as pd
import scipy.io as sio

import pywt
import scipy.stats

from collections import defaultdict, Counter

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

# 数据集中有35个ARR（心率失常），30个CHF（心力衰竭），36个NSR（窦性心率正常）


class Wavelet(object):
    def __init__(self, array):
        self.array = array

    def __del__(self):
        class_name = self.__class__.__name__

    def get_entropy(self):  # 通过概率值计算熵，作为信号复杂度的度量
        counter_set = Counter(self.array).most_common()
        # collection包中Counter函数的作用：把每个元素及其出现的次数进行统计。
        # 例: Counter('daddy')输出为Counter({'d':3, 'a':1, 'y':1})
        # Counter.most_common则是把Counter的输出结果转化为数组，且将各个数出现的顺序按照"出现次数从多到少"排列。
        possibilities = []
        for item in counter_set:
            possibilities.append(item[1]/len(self.array))
        entropy = scipy.stats.entropy(possibilities)
        return entropy

    def common_statistics(self):  # 获取小波变换后信号几个常见的统计量
        # np.nanpercentile计算百分比分位数。
        per5 = np.nanpercentile(self.array, 5)  # 5%分位
        per25 = np.nanpercentile(self.array, 25)
        per75 = np.nanpercentile(self.array, 75)
        per95 = np.nanpercentile(self.array, 95)
        per50 = np.nanpercentile(self.array, 50)  # 中位数
        mean = np.nanmean(self.array)
        std = np.nanstd(self.array)
        var = np.nanvar(self.array)
        rms = np.nanmean(np.sqrt(self.array**2))  # 均方根
        return [per5, per25, per75, per95, per50, mean, std, var, rms]

    def cross_times(self):
        cross_zero = np.nonzero(np.diff(np.array(self.array) > 0))[0]
        # np.diff对数组做差分(True和False差分时视为0，1)，np.nonzero取出数组非零值的下标
        zero_cross_time = len(cross_zero)
        # 过零次数，即信号穿过y轴的次数
        cross_mean = np.nonzero(np.diff(np.array(self.array) > np.nanmean(self.array)))[0]
        mean_cross_time = len(cross_mean)
        # 过均值次数，即信号穿越y=平均值这一直线的次数
        return [zero_cross_time, mean_cross_time]

    def get_features(self):
        ent = self.get_entropy()
        cross = self.cross_times()
        stat = self.common_statistics()
        return [ent] + cross + stat
    # 特征：方差、标准差、平均值、中位数、下四分位数、上四分位数、5%分位数、95%分位数、均方根、熵、过零次数、过均值次数


# =====================================================三种方法公共部分=====================================================
def get_train_test(df, y_col, x_cols, ratio):  # 将数据按比例切分为训练集、测试集
    np.random.seed(0)  # 随机种子
    region = np.random.rand(len(df)) < ratio
    df_train = df[region]
    df_test = df[~region]
    Y_train = df_train[y_col].values
    Y_test = df_test[y_col].values
    X_train = df_train[x_cols].values
    X_test = df_test[x_cols].values
    return df_train, df_test, X_train, Y_train, X_test, Y_test


# 读取数据（这里原数据集保存在.mat文件中，用scipy.sio.loadmat读取）
filename = './balanced_data.mat'
ecg_data = sio.loadmat(filename)  # ecg_data中包括信号和标签两部分
ecg_signals = ecg_data['ECGData'][0][0][0]  # ecg_data中的信号
ecg_labels_ = ecg_data['ECGData'][0][0][1]  # ecg—data中的标签
ecg_labels = list(map(lambda x: x[0][0], ecg_labels_))  # 将label提取出来作为输入分类器的标签
# print(ecg_labels)
dict_ecg_data = defaultdict(list)  # 构建一个字典，将信号和标签对应起来
for ii, label in enumerate(ecg_labels):  # ii为序号，label为标签值
    dict_ecg_data[label].append(ecg_signals[ii])

list_labels = []
list_features = []
for label, signals in dict_ecg_data.items():
    y_value = list(dict_ecg_data.keys()).index(label)  # 把y的字符串标签转换为数字1-3
    for signal in signals:  # v中是同一种标签对应的所有信号，因此signal in v代表同种类型的所有信号。signal代表单个信号。
        features = []
        list_labels.append(y_value)
        #print('%%%',signal)
        a=3
        list_wavelet = pywt.wavedec(signal, 'sym5', level=a)  # 用sym5小波实现3阶DWT
        # list_wavelet是wavedec函数返回的系数数组，
        for list_value in list_wavelet:  # 对每个子带执行以下操作
            Wave = Wavelet(list_value)  # 从类Wavelet中实现特征提取
            features += Wave.get_features()
            del Wave  # 销毁对象
        list_features.append(features)
df = pd.DataFrame(list_features)
ycol = 'y'
xcols = list(range(df.shape[1]))
df.loc[:,ycol] = list_labels
df_train, df_test, X_train, Y_train, X_test, Y_test = get_train_test(df, ycol, xcols, ratio = 0.7)
cls = GradientBoostingClassifier(n_estimators=2000)
cls.fit(X_train, Y_train)
train_score1 = cls.score(X_train, Y_train)
test_score1 = cls.score(X_test, Y_test)
print("When a = {}, The Train Score of GBDT is {}".format(a,train_score1))
print("When a = {}, The Test Score of GBDT is {}".format(a,test_score1))
clf = RandomForestClassifier(n_estimators=2000)
clf.fit(X_train, Y_train)
train_score2 = clf.score(X_train, Y_train)
test_score2 = clf.score(X_test, Y_test)
print("When a = {}, The Train Score of RF is {}".format(a,train_score2))
print("When a = {}, The Test Score of RF is {}".format(a,test_score2))
mlp = MLPClassifier(max_iter=1000)
mlp.fit(X_train, Y_train)
train_score3 = mlp.score(X_train, Y_train)
test_score3 = mlp.score(X_test, Y_test)
print("When a = {}, The Train Score of MLP is {}".format(a,train_score3))
print("When a = {}, The Test Score of MLP is {}".format(a,test_score3))
svm = SVC()
svm.fit(X_train, Y_train)
train_score4 = svm.score(X_train, Y_train)
test_score4 = svm.score(X_test, Y_test)
print("When a = {}, The Train Score of SVM is {}".format(a,train_score4))
print("When a = {}, The Test Score of SVM is {}".format(a,test_score4))
nb = GaussianNB()
nb.fit(X_train, Y_train)
train_score5 = nb.score(X_train, Y_train)
test_score5 = nb.score(X_test, Y_test)
print("When a = {}, The Train Score of Naive Bayes is {}".format(a,train_score5))
print("When a = {}, The Test Score of Naive Bayes is {}".format(a,test_score5))
scores=[test_score1,test_score2,test_score3,test_score4,test_score5]
max_score=max(scores)
print("Symlets小波变换分解阶数为{}阶时，常见分类器最高准确率为{}".format(a,max_score))