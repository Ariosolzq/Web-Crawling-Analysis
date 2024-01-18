import numpy as np
import pandas as pd

# 1.读取数据集
# df = pd.read_csv(r'C:\Users\76044\Desktop\file.csv',encoding='gbk')
df = pd.read_csv('file.csv', encoding='gbk')

# 2.检查数据集，了解基本情况,统计性描述
pd.set_option('display.width', None)
print('数据条数：', '\n', len(df), '\n')
print('前20项：', '\n', df.head(20), '\n')
print('基本描述：', '\n', df.describe(), '\n')
print('数据集规模：', df.shape)
print(df.info(), '\n')
collist = df.keys()
print('数据集列名：\n', collist)
print('数据集空值数：\n', df.isnull().sum())
# 检查结果：无空值，发现object数据类型

# 3.数据可视化处理
# 二维可视化方法如折线图、饼状图、条形图、直方图、散点图、柱状图、等高图、热力图等
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.pyplot import MultipleLocator
from pylab import mpl

# 解决中文显示问题
plt.rcParams['font.sans-serif'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False

# 指定默认字体，解决保存图像是负号'-'显示为方块的问题
mpl.rcParams['font.sans-serif'] = ['FangSong']
mpl.rcParams['axes.unicode_minus'] = False

# 3.1 不同户型的房屋个数
housetype = df['户型'].value_counts()# 计算户型的所占的个数
asd,sdf = plt.subplots(1,1,dpi=100) #获取前10条数据
housetype.head(5).plot(kind='bar',x='户型',y='数量',title='户型数量分布',ax=sdf)
plt.xticks(rotation=30)# 横坐标倾斜
plt.xlabel('房屋个数')
plt.ylabel('户型')
plt.legend(['数量'])
plt.savefig('户型数量分布.jpg')
plt.show()

# 3.1.5 绘制房源户型分布条形图
lis = []
for key in df['户型'].value_counts():
    lis.append(key)
plt.figure(figsize=(12,10),dpi=80)
lis = plt.barh(range(len(lis)),lis,height=0.6,color='cyan',alpha=0.7)
plt.yticks(range(len(lis)),df['户型'].value_counts().index)
plt.xlim(0,2000)
plt.xlabel('人数')
plt.title('户型数量分布')
plt.grid(alpha=0.2)
# 为条形图添加数据标签
for x in lis:
    width = x.get_width()
    plt.text(width+1,x.get_y()+x.get_height()/2,str(width),ha='center',va='center')
plt.savefig('户型人数显示.jpg')
plt.show()

# 3.2 大众对户型的关注及偏好
type_interest_group = df['关注人数'].groupby(df['户型']).agg([('户型总数', 'count'), ('关注人数', 'sum')])
# 获取户型数量大于50 的数据
ti_sort = type_interest_group[type_interest_group['户型总数'] > 50 ].sort_values(by='户型总数')
asd,sdf = plt.subplots(1,1,dpi=150)
ti_sort.plot(kind='barh',alpha=0.7,grid=True,ax=sdf)
plt.title('二手房户型和关注人数分布')
plt.xlabel('关注人数')
plt.ylabel('户型')
plt.xticks(rotation=30)# 横坐标倾斜
plt.savefig('二手房户型和关注人数分布.jpg')
plt.show()

# 3.3 户型和面积的关系
area_level = [0, 50, 100, 150, 200, 250, 300, 500]
label_level = ['小于50', '50-100', '100-150', '150-200', '200-250', '250-300', '300-350']
# 算出所面积在哪个区间
df['面积'] = df['面积'].astype(float)
are_cut = pd.cut(df['面积'],area_level,label_level)
# 计算面积在所设置的区间的数量
acreage = are_cut.value_counts()
asd,sdf = plt.subplots(1,1,dpi=150)
acreage.plot(kind='bar',rot=30,alpha=0.4,grid=True,ax=sdf) # 倾斜、网格线
# values=[]
# for i in acreage.values:
#     values.append(i)
# print(values)
# for a, b ,c in zip(area_level, label_level,values):
#     plt.text(a, b, c , ha='center', va='bottom', fontsize=17)  # fontsize表示柱坐标上显示字体的大小
plt.title('二手房面积分布')
plt.xlabel('面积')
plt.legend(['数量'])
plt.savefig('二手房面积分布.jpg')
plt.show()

# 3.4 各个行政区房源均价
region =df.groupby('行政区').mean(numeric_only=True)['平方米价格']
asd,sdf = plt.subplots(1,1,dpi=150)
region.plot(kind='bar',x='region',y='unitprice', title='各个区域房源均价',ax=sdf)
plt.xticks(rotation=30)# 横坐标倾斜
plt.savefig('各个区域房源均价.jpg')
plt.show()

# 3.5各个区域房源数量排序
region_num = df.groupby('行政区').size().sort_values(ascending=False)
asd,sdf = plt.subplots(1,1,dpi=150)
region_num.plot(kind='bar',x='region',y='size',title='各个区域房源数量分布',ax=sdf)
plt.legend(['数量'])
plt.savefig('各个区域房源数量分布.jpg')
plt.show()

# 3.6各个区域小区房源数量
community_num =df.groupby('小区').size().sort_values(ascending=False) #画图
asd,sdf = plt.subplots(1,1,dpi=150) #取前十五个
community_num.head(15).plot(kind='bar',x='xiaoqu',y='size',title='各个区域小区房源数量',ax=sdf)
plt.legend(['数量'])
plt.xticks(rotation=30,fontsize = 5)# 横坐标倾斜，调整字体
plt.savefig('各个区域小区房源数量.jpg')
plt.show()

#3.7 各个小区的房源均价
community_mean = df.groupby('小区').mean(numeric_only=True)['平方米价格'].sort_values(ascending=False) #画图
asd,sdf = plt.subplots(1,1,dpi=150) #前10 条
community_mean.head(10).plot(kind='bar',x='小区',y='mean',title='各个小区房源均价',ax=sdf)
plt.legend(['均价'])
plt.xticks(rotation=30,fontsize = 5)# 横坐标倾斜，调整字体
plt.savefig('各个小区房源均价.jpg')
plt.show()

# 3.8散点图：数据量较大时，散点图保留特征
plt.figure(figsize=(12, 5))
x=df['平方米价格']
plt.xlabel('平方米价格', fontsize=15)
plt.ylabel('面积', fontsize=0)
y=df['面积']
plt.scatter(x, y)
plt.title('面积与价格的散点图')
plt.show()

plt.xlabel('平方米价格', fontsize=15)
plt.ylabel('行政区', fontsize=0)
y=df['行政区']
plt.scatter(x, y)
plt.title('行政区与价格的散点图')
plt.show()

plt.xlabel('平方米价格', fontsize=15)
plt.ylabel('装修类型', fontsize=0)
y=df['装修类型']
plt.scatter(x, y)
plt.title('装修类型与价格的散点图')
plt.show()

# 3.9词云-房屋标签
plt.rcParams["font.sans-serif"]=["SimHei"]
from wordcloud import WordCloud
plt.subplots(figsize=(15,15))
wordcloud = WordCloud(font_path='./simsun.ttc',background_color='white',width=1920,height=1080).generate(" ".join(df.房屋标签))
plt.imshow(wordcloud)
plt.axis('off')
plt.show()

# 3.9词云-社区
plt.subplots(figsize=(15,15))
wordcloud = WordCloud(font_path='simsun.ttc',background_color='white',width=1920,height=1080).generate(" ".join(df.社区))
plt.imshow(wordcloud)
plt.axis('off')
plt.show()


# 4.数据再处理，准备进行数据分析与挖掘
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
df.drop(['房屋标签','小区','朝向'], axis=1, inplace=True)
df.drop(['总价'], axis=1, inplace=True)  # 此处删除总价，后续可以更改为删除平方米价格，选其一作为预测目标
# df.drop(['平方米价格'],axis=1, inplace=True)
print(df.head())

# 4.1文本类型转编码
def tran(df):
    for column in df.columns[df.columns.isin(['小区', '社区', '行政区','户型', '行政区','装修类型'])]:
        df[column] = df[column].factorize()[0]
    return df

df = tran(df.copy())
print(df)

# 4.2 划分数据集，特征工程
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
x_features = list(df.keys())
x_features.remove('平方米价格')
y_feature = ['平方米价格']
df_x = df[x_features]
df_y = df[y_feature]

# 对数据进行归一化处理
mms = MinMaxScaler()
df_x = mms.fit_transform(df_x)
df_y = mms.fit_transform(df_y.values.reshape(-1,1))

# 对数据进行标准化处理
ss = StandardScaler()
df_x = ss.fit_transform(df_x)
df_y = ss.fit_transform(df_y)

# 划分训练集、测试集
X_train, X_test, y_train, y_test = train_test_split(df_x, df_y,test_size=0.3, random_state=42)

# # 热力图
sns.heatmap(df.corr(),cmap='flare',annot=True)
plt.show()

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