import scipy.io as sio
from collections import defaultdict
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
import numpy as np
from scipy import stats
from scipy.signal import filtfilt, butter
import matplotlib.pyplot as plt


class getFeature():
    def calculate_features(self, inputs):
        inputs = np.array(inputs)
        # 最小值
        min = np.min(inputs)
        # 最大值
        max = np.max(inputs)
        # 5%分位数
        per5=np.nanpercentile(inputs, 5)
        # 95%分位数
        per95=np.nanpercentile(inputs, 95)
        # 上四分位数
        per25 = np.nanpercentile(inputs, 25)
        # 下四分位数
        per75 = np.nanpercentile(inputs, 75)
        # 均值
        mean = np.mean(inputs)
        # 中值
        median = np.median(inputs)
        # 中值绝对偏差
        mad = stats.median_absolute_deviation(inputs)
        # 标准差
        std = np.std(inputs, ddof=1)
        # 偏度
        skew = stats.skew(inputs)
        # 峰度
        kurtosis = stats.kurtosis(inputs)
        # 四分位数范围
        iqr = stats.iqr(inputs)
        # 过零率
        cross_zero = np.nonzero(np.diff(np.array(inputs) > 0))[0]
        pass_zero=len(cross_zero)/len(inputs)*10
        # 过均值率
        cross_mean = np.nonzero(np.diff(np.array(inputs) > np.nanmean(inputs)))[0]
        pass_mean=len(cross_mean)/len(inputs)*10
        # 频域偏度系数
        wskew = stats.skew(inputs)
        # 频域峰度系数
        wkurtosis = stats.kurtosis(inputs)
        # 将所有特征合并为数组
        array = [min, max, mean, median, mad, std, skew, kurtosis, iqr, per5, per25, per75, per95, pass_zero, pass_mean, wskew, wkurtosis]
        return array


def get_train_test(df, y_col, x_cols, ratio):  # 将数据按比例切分为训练集、测试集
    region = np.random.rand(len(df)) < ratio
    df_train = df[region]
    df_test = df[~region]
    Y_train = df_train[y_col].values
    Y_test = df_test[y_col].values
    X_train = df_train[x_cols].values
    X_test = df_test[x_cols].values
    return df_train, df_test, X_train, Y_train, X_test, Y_test


def medfit(vector, n_odd):  # 中值滤波
    odd=int((n_odd-1)/2)
    fitted=[0]*len(vector)
    for i in range(0,odd):
        fitted[i]=vector[i]
    for i in range(len(vector)-odd,len(vector)):
        fitted[i]=vector[i]
    for i in range(odd,len(vector)-odd):
        obj=vector[i-odd:i+odd+1]
        obj=sorted(obj)
        fitted[i]=obj[odd]
    return fitted


def butterworth(vector, N, Wn):
    b, a = butter(N, Wn, btype='lowpass', output='ba')
    arr = filtfilt(b, a, vector)
    return arr


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
raw_data = []
# 把ecg数据集转为一个矩阵，每行是一个信号及其标签。
for label, signals in dict_ecg_data.items():
    y_value = list(dict_ecg_data.keys()).index(label)  # 把y的字符串标签转换为数字1-3
    plot=signals[0]
    x=np.linspace(0,199,num=200)
    y=signals[0][0:200]
    plt.plot(x, y, "g-", label="Original")
    plot=medfit(plot,5)
    plot1=butterworth(plot, 8, 1/9)
    y2=plot[0:200]
    plt.plot(x, y2, "b-", label="After")
    plt.legend()
    plt.show()
    for signal in signals:  # v中是同一种标签对应的所有信号，因此signal in v代表同种类型的所有信号。signal代表单个信号。
        signa=signal.tolist()
        sign=medfit(signa,5)  # 中值滤波
        sig=butterworth(sign, 8, 1/9)  # 四阶巴特沃斯低通数字滤波器
        si=sig.tolist()
        si.append(y_value)
        raw_data.append(si)
# PCA降维，确定5个最优的统计量作为特征
FEA=getFeature()
final_data=[]
for line in raw_data:
    features=FEA.calculate_features(line)
    features.append(line[-1])
    final_data.append(features)
X,Y=list(),list()
for i in range(int(len(final_data))):
    X.append(final_data[i][0:len(final_data[i])-1])
    Y.append(final_data[i][-1])
pca=PCA(n_components=9)
new_data=pca.fit_transform(X)
together=[]
for i in range(len(new_data)):
    a=new_data[i].tolist()
    a.append(Y[i])
    together.append(a)
np.random.seed(1)
np.random.shuffle(together)
X_train,Y_train=list(),list()
X_test,Y_test=list(),list()
for i in range(int(len(together)*0.7)):
    X_train.append(together[i][0:len(together[i])-1])
    Y_train.append(together[i][-1])
for i in range(int(len(together)*0.7),len(together)):
    X_test.append(together[i][0:len(together[i])-1])
    Y_test.append(together[i][-1])
#=======================================================================随机森林========================================================================================================================
clf=GradientBoostingClassifier(n_estimators=1000)
clf.fit(X_train, Y_train)
print("GBDT Accuracy on training set is : {}".format(clf.score(X_train, Y_train)))
print("GBDT Accuracy on test set is : {}".format(clf.score(X_test, Y_test)))
clf=RandomForestClassifier(n_estimators=1000)
clf.fit(X_train, Y_train)
print("RF Accuracy on training set is : {}".format(clf.score(X_train, Y_train)))
print("RF Accuracy on test set is : {}".format(clf.score(X_test, Y_test)))
# print(classification_report(Y_test, Y_test_pred))
clf=MLPClassifier(max_iter=2000)
clf.fit(X_train, Y_train)
print("MLP Accuracy on training set is : {}".format(clf.score(X_train, Y_train)))
print("MLP Accuracy on test set is : {}".format(clf.score(X_test, Y_test)))

clf = GaussianNB()
clf = clf.fit(X_train, Y_train)
print("NB Accuracy on training set is : {}".format(clf.score(X_train, Y_train)))
print("NB Accuracy on test set is : {}".format(clf.score(X_test, Y_test)))

svm = SVC()
svm = svm.fit(X_train,Y_train)
print("SVM Accuracy on training set is : {}".format(svm.score(X_train, Y_train)))
print("SVM Accuracy on test set is : {}".format(svm.score(X_test, Y_test)))
