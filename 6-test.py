import numpy as np
import pandas as pd

# 1.读取数据集
df = pd.read_csv(r'C:\Users\76044\Desktop\listings.csv')

# 2.检查数据集，了解基本情况
pd.set_option('display.width',None)
print('数据条数：','\n',len(df),'\n')
print('前20项：','\n',df.head(20),'\n')
print('基本描述：','\n',df.describe(),'\n')
print('数据集规模：',df.shape)
print(df.info(), '\n')
collist=df.keys()
print('数据集列名：\n',collist)
print('数据集空值数：\n',df.isnull().sum())
# 检查结果：

# 3.1 对数据记录进行详细地分析与处理
# 删除无用列、检查缺失值
df.drop(['id','host_id','license'],axis=1,inplace=True)
df_name=df[df['name'].isnull()]
print(df_name.head())

df_host_name=df[df['host_name'].isnull()]
print(df_host_name.head())

print('前5项：','\n',df.head(5),'\n')
print(df.info(), '\n')
collist=df.keys()
print('数据集列名：\n',collist)
print('数据集空值数：\n',df.isnull().sum())

# 没有被评论过
df_null=df[df['last_review'].isnull()]
print('没有被评论过',df_null)

# 对Nan进行处理，改为0
df['last_review'].fillna(0,inplace=True)
df['reviews_per_month'].fillna(0,inplace=True)

# 处理价格为0的记录
df_price0=df.loc[df.price==0,'price']
print(df_price0)
df=df[df['price']!=0]
print(df.describe())

# 检查object数据类型的独一性
for i in range(len(collist)):
    if df[collist[i]].dtype == object:
        # 统计数值个数
        num=df[collist[i]].nunique()
        print('\n列名：{} ；不同值个数：{} '.format(collist[i],num))
        print(df[collist[i]].unique(),'\n')
    else:
        continue

# 3.2数据可视化处理
# 二维可视化方法如折线图、饼状图、条形图、直方图、散点图、柱状图、等高图、热力图等
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.pyplot import MultipleLocator

# 解决中文显示问题
plt.rcParams['font.sans-serif'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False

# 查询行政区内的房屋数
print('\n行政区内的房屋数：')
print(df['neighbourhood_group'].value_counts())
plt.figure(figsize=(16,13),dpi=100)
sns.scatterplot(x=df.longitude,y=df.latitude, hue=df.neighbourhood_group)
plt.title("行政区内的房屋散点分布")
plt.grid('both')
plt.style.use('fivethirtyeight')
plt.show()

# 查询房屋类型，先数据归类
print('\n房屋类型：')
# 散点图
print(df['room_type'].value_counts())
plt.figure(figsize=(16,13),dpi=100)
sns.scatterplot(x=df.longitude,y=df.latitude, hue=df.room_type)
plt.title("纽约房屋种类散点分布")
plt.grid('both')
plt.style.use('fivethirtyeight')
plt.show()
# 条形图
plt.figure(figsize=(16,13),dpi=100)
plt.title("纽约房屋种类条形图统计")
sns.countplot(x=df["room_type"])
plt.grid('both')
plt.style.use('fivethirtyeight')
plt.show()

# 房价分析
print('房价分析',df.groupby(by='room_type')['price'].describe())
# # 使用plotly
# import plotly.express as px
# fig = px.scatter(df, x="longitude", y="latitude", color='price', width=1500, height=900, hover_data=['name'])
# fig.show()
plt.figure(figsize=(16,13),dpi=100)
sns.scatterplot(x=df.longitude,y=df.latitude, hue=df.price)
plt.title("纽约房屋价格散点分布")
plt.grid('both')
plt.style.use('fivethirtyeight')
plt.show()
# 条形图
plt.figure(figsize=(16,13),dpi=100)
plt.title("纽约房屋价格条形图统计")
x = MultipleLocator(200)
sns.countplot(x=df["price"])
plt.grid('both')
plt.style.use('fivethirtyeight')
x_major_locator=MultipleLocator(100)
ax=plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
plt.show()



# 词云
from wordcloud import WordCloud
plt.subplots(figsize=(15,15))
wordcloud = WordCloud(background_color='white',width=1920,height=1080).generate(" ".join(df.neighbourhood))
plt.imshow(wordcloud)
plt.axis('off')
plt.show()

# 3.数据处理
df.drop(['name','host_name','neighbourhood','last_review'], axis=1, inplace=True)
print(df.head())

# from sklearn.preprocessing import OneHotEncoder as ohe
# 也可以使用onehot方法
# oh_encoder = ohe(handle_unknown='ignore', dtype=int)
# df_encoded = pd.DataFrame(oh_encoder.fit_transform(df[['neighbourhood_group','room_type']]).toarray())
# print(df_encoded)
#
# df_treated = pd.concat([df.reset_index(drop=True),df_encoded.reset_index(drop=True)], axis=1)
# df_treated.drop(['neighbourhood_group','room_type'], axis=1, inplace=True)
# print(df_treated)

def tran(df):
    for column in df.columns[df.columns.isin(['neighbourhood_group', 'room_type'])]:
        df[column] = df[column].factorize()[0]
    return df

df = tran(df.copy())
print(df)

# 4.数据分析准备
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

x_features = list(df.keys())
x_features.remove('price')
y_feature = ['price']
df_x = df[x_features]
df_y = df[y_feature]

print(df_x)
print(df_y)

# # 对数据进行归一化处理
# mms = MinMaxScaler()
# df_x = mms.fit_transform(df_x)
# df_y = mms.fit_transform(df_y.values.reshape(-1,1))
#
# 对数据进行标准化处理
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
print('\n线性回归')
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
print('\n回归决策树')
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
print('\n随机森林回归分析')
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


