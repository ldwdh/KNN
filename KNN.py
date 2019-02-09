from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
#读取数据，数据采用sklearn自带的数据集
digits=load_digits()
data=digits.data
#分割数据，数据集中25%作为测试数据，其余作为训练数据
train_x,test_x,train_y,test_y=train_test_split(data,digits.target,test_size=0.25,random_state=1)
#采用z-score规范化
ss=preprocessing.StandardScaler()
train_ss_x=ss.fit_transform(train_x)
test_ss_x=ss.transform(test_x)
#创建KNN分类器
knn=KNeighborsClassifier()
knn.fit(train_ss_x,train_y)
#预测结果并输出
predict_y=knn.predict(test_ss_x)
print('KNN准确率:{:.4f}'.format(accuracy_score(predict_y,test_y)))
#创建SVM分类器
svm=SVC()
svm.fit(train_ss_x,train_y)
predict_y=svm.predict(test_ss_x)
print('SVM准确率:{:.4f}'.format(accuracy_score(predict_y,test_y)))
#采用Min-Max规范化
mm=preprocessing.MinMaxScaler()
train_mm_x=mm.fit_transform(train_x)
test_mm_x=mm.transform(test_x)
#创建朴素贝叶斯分类器
mnb=MultinomialNB()
mnb.fit(train_mm_x,train_y)
predict_y=mnb.predict(test_mm_x)
print('多项式朴素贝叶斯准确率:{:.4f}'.format(accuracy_score(predict_y,test_y)))
#创建cart决策树分类器
dtc=DecisionTreeClassifier()
dtc.fit(train_ss_x,train_y)
predict_y=dtc.predict(test_ss_x)
print('cart决策树准确率:{:.4f}'.format(accuracy_score(predict_y,test_y)))