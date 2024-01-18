import numpy as np
import pandas as pd

# 1.读取数据集
df = pd.read_csv(r'C:\Users\76044\Desktop\sleeping.csv')

# 2.检查数据集，了解基本情况
pd.set_option('display.width',None)
print('前20项：','\n',df.head(20),'\n')
print('基本描述：','\n',df.describe(),'\n')
print(df.info(), '\n')
# 检查结果：无空值，发现object数据类型

# 2.1 对数据记录进行详细地分析与处理
collist=df.keys()
print(collist)

# 对数据列名进行简化
df.rename(columns = {'Avg hrs per day sleeping':'Hours'}, inplace = True)
df.rename(columns = {'Standard Error':'Error'}, inplace = True)
df.rename(columns = {'Type of Days':'Days'}, inplace = True)
df.rename(columns = {'Age Group':'Age'}, inplace = True)
collist=df.keys()
print(collist)

# 检查object数据类型的独一性
for i in range(len(collist)):
    if df[collist[i]].dtype == object:
        # 统计数值个数
        num=df[collist[i]].nunique()
        print('列名：{} ；不同值个数：{} '.format(collist[i],num))
        print(df[collist[i]].unique(),'\n')
    else:
        continue

# 根据数据集的物理意义与检查结果，对index列、Period列和Activity列进行删除处理
df.drop(['index','Period','Activity'],axis=1,inplace=True)
print(df.head(20))
print(df.info())

# 2.2数据可视化处理
# 二维可视化方法如折线图、饼状图、条形图、直方图、散点图、柱状图、等高图、热力图等
import matplotlib.pyplot as plt
import seaborn as sns

# 解决中文显示问题
plt.rcParams['font.sans-serif'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False

# 折线图：年份与睡眠时间的关系
plt.figure(figsize=(8, 8))
group_year=df.groupby('Year')
Y=group_year['Hours'].mean()
plt.xticks(np.arange(2003,2018))
plt.xlabel("Year")
plt.ylabel("Average Hours")
plt.title("平均睡眠时间变化折线图")
Y.plot(kind='line')
plt.show()

# 散点图：数据量较大，散点图保留特征
plt.figure(figsize=(12, 5))
x=df['Hours']
plt.xlabel('Hours', fontsize=15)
plt.title('睡眠时间与日期、年龄、性别的散点图')
for i in ['Days','Age','Sex']:
    plt.ylabel(i, fontsize=0)
    plt.scatter(x, df[i])
plt.show()

# 热力图
sns.heatmap(df.corr(),cmap='flare',annot=True)
plt.show()

# 标准差与平均睡眠时间组合图
sns.set_palette('summer')
sns.jointplot(kind='reg',y='Hours',x='Error',data=df)
plt.show()

# 3.数据处理
# 发现Days中的all days类型和Sex中的Both数据没有较大的实际意义，考虑将其剔除
df=df[df['Days']!='All days']
df=df[df['Sex']!='Both']
# print(df.head(20))
# print(df.describe())
# print(df.info())

# Get Dummies方法
# df = pd.concat([df,pd.get_dummies(df[['Days','Sex','Age']],drop_first=True)],axis=1)
# df.drop(['Days','Age','Sex'],axis=1,inplace=True)
# print(df.describe())
# print(df.head(20))

# one-hot方法提取文本特征
from sklearn.preprocessing import OneHotEncoder as ohe

oh_encoder = ohe(handle_unknown='ignore', dtype=int)
df_encoded = pd.DataFrame(oh_encoder.fit_transform(df[['Days','Age','Sex']]).toarray())
print(df_encoded)

# 编码后的数据集与原始数据集合并
df_treated = pd.concat([df.reset_index(drop=True),df_encoded.reset_index(drop=True)], axis=1)
df_treated.drop(['Days','Age','Sex'], axis=1, inplace=True)
print(df_treated)

# 4.数据分析准备
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from sklearn import metrics
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

x_features = list(df_treated.keys())
x_features.remove('Hours')
y_feature = ['Hours']
df_x = df_treated[x_features]
df_y = df_treated[y_feature]

print(df_x)
print(df_y)

# 对数据进行归一化处理
mms = MinMaxScaler()
df_x = mms.fit_transform(df_x)
df_y = mms.fit_transform(df_y.values.reshape(-1,1))

# # 对数据进行标准化处理
# ss = StandardScaler()
# df_x = ss.fit_transform(df_x)
# df_y = ss.fit_transform(df_y)

# 划分训练集、测试集
X_train, X_test, y_train, y_test = train_test_split(df_x, df_y,test_size=0.3, random_state=42)

# 5.线性回归LinearRegression
from sklearn.linear_model import LinearRegression
lr_model = LinearRegression()
lr_model.fit(X_train,y_train)
a = lr_model.coef_
print("a: ", a)
y_pred=lr_model.predict(X_test)

# 5.2模型评估
print("原始数据准确率：", lr_model.score(df_x,df_y))
print("训练集准确率：", lr_model.score(X_train,y_train))
print("测试集准确率：", lr_model.score(X_test,y_test))
print('剩余标准差Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print("决定系数R2 Score:",r2_score(y_test, y_pred))
print("平均绝对误差Mean Absolute Error:",mean_absolute_error(y_test, y_pred))
print("平均平方误差Mean Squared Error:",mean_squared_error(y_test, y_pred))

# 5.3画图
plt.plot(list(range(0,len(X_test))),y_test,marker='o')
plt.plot(list(range(0,len(X_test))),y_pred,marker='*')
plt.legend(['真实值','预测值'])
plt.title('线性回归预测值与真实值的对比')
plt.show()

# 6.岭回归Ridge Regression Model
from sklearn.linear_model import Ridge

def ridge_regression(X_train, X_test, y_train, y_test, alpha):
    ridgereg = Ridge(alpha=alpha, normalize=True)
    ridgereg.fit(X_train, y_train)
    y_pred = ridgereg.predict(X_test)

    print("\n岭回归系数a为", str(alpha))
    print("岭回归截距：", ridgereg.intercept_)
    print("岭回归各项系数",ridgereg.coef_)
    print('剩余标准差Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    print("决定系数R2 Score:",r2_score(y_test, y_pred))
    print("平均绝对误差Mean Absolute Error:",mean_absolute_error(y_test, y_pred))
    print("平均平方误差Mean Squared Error:",mean_squared_error(y_test, y_pred))

    plt.plot(list(range(0, len(X_test))), y_test, marker='o')
    plt.plot(list(range(0, len(X_test))), y_pred, marker='*')
    plt.legend(['真实值', '预测值'])
    plt.title("预测值与真实值的对比")
    plt.show()

alphaValues = [0,  0.2, 0.5, 0.7]
for i in range(0, len(alphaValues)):
    ridge_regression(X_train, X_test, y_train, y_test,alphaValues[i])

# 7.决策树回归DecisionTreeRegressor
# 7.1模型拟合
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state=42).fit(X_train,y_train)
y_pred=regressor.predict(X_test)

# 7.2模型评估
print("原始数据准确率：", regressor.score(df_x,df_y))
print("训练集准确率：", regressor.score(X_train,y_train))
print("测试集准确率：", regressor.score(X_test,y_test))
print('剩余标准差Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print("决定系数R2 Score:",r2_score(y_test, y_pred))
print("平均绝对误差Mean Absolute Error:",mean_absolute_error(y_test, y_pred))
print("平均平方误差Mean Squared Error:",mean_squared_error(y_test, y_pred))

# 7.3画图
plt.plot(list(range(0,len(X_test))),y_test,marker='o')
plt.plot(list(range(0,len(X_test))),y_pred,marker='*')
plt.legend(['真实值','预测值'])
plt.title('决策回归树预测值与真实值的对比')
plt.show()


# 8.随机森林回归预测RandomForestRegressor
# 8.1模型拟合
from sklearn.ensemble import RandomForestRegressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, np.ravel(y_train))
# rf_model.fit(x,np.ravel(y))
# warning，因为随机森林的模型对于y项数据有要求，调用ravel(y)展平数据矩阵.
y_pred=rf_model.predict(X_test)

# 8.2模型评估
print("原始数据准确率：", rf_model.score(df_x,df_y))
print("训练集准确率：", rf_model.score(X_train,y_train))
print("测试集准确率：", rf_model.score(X_test,y_test))
print('剩余标准差Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print("决定系数R2 Score:",r2_score(y_test, y_pred))
print("平均绝对误差Mean Absolute Error:",mean_absolute_error(y_test, y_pred))
print("平均平方误差Mean Squared Error:",mean_squared_error(y_test, y_pred))

# 8.3画图
plt.plot(list(range(0,len(X_test))),y_test,marker='o')
plt.plot(list(range(0,len(X_test))),y_pred,marker='*')
plt.legend(['真实值','预测值'])
plt.title('随机森林回归预测值与真实值的对比')
plt.show()
