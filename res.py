#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[9]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv('光伏发电装机容量浙江累计值.csv',parse_dates=['date'])

# 假设你的 DataFrame 叫 df，其中有 'date' 和 'capacity_GW' 列
# 将 'date' 列转换为日期时间格式
df['date'] = pd.to_datetime(df['date'])

# 格式化日期为年-月的字符串格式
df['year_month'] = df['date'].dt.strftime('%Y-%m')

# 使用 Pandas 的绘图功能
plt.figure(figsize=(20, 6))  # 设置图形大小
plt.plot(df['date'], df['capacity_GW'], marker='o', linestyle='-')

# 设置x轴标签为年-月的格式
plt.xticks(df['date'], df['year_month'])

# 设置标签和标题
plt.xlabel('Date (Year-Month)')
plt.ylabel('Capacity (Wkw)')
plt.title('Solar Capacity Over Time')

# 自动调整日期标签
plt.gcf().autofmt_xdate()

# 显示图形
plt.show()


# In[10]:


# capacity单位 万千瓦

# In[32]:


N=4
feas_name=['capacity_GW']
for i in range(1,N):
    df[f'capacity-{i}']=df['capacity_GW'].shift(i)
    feas_name.append(f'capacity-{i}')
df['y']=df['capacity_GW'].shift(-1)
df=df.dropna()
df


# In[57]:


from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_percentage_error


# In[58]:


x=df[feas_name].values
y=df['y'].values
# 划分数据集为训练集和验证集
from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2,shuffle=False)


# In[73]:


# 初始化MLP分类器
mlp = MLPRegressor(hidden_layer_sizes=(5, 5), max_iter=1000, random_state=42)
# 使用训练集来训练MLP模型
mlp.fit(train_x, train_y)
# 使用训练好的模型进行预测
pred_y = mlp.predict(test_x)
# 计算模型在测试集上的准确率
accuracy = 1-mean_absolute_percentage_error(test_y, pred_y)
print(f"Accuracy on test set: {accuracy:.2f}")
print('pred_y:',pred_y)
print('test_y:',test_y)


# In[ ]:
stack=list(df['capacity_GW'].iloc[-N:])
res=[]
start_date = '2024-09-01'
end_date = '2035-12-31'

date_range = pd.date_range(start=start_date, end=end_date, freq='3M',closed='right')
pred_steps=len(date_range)
for step in range(pred_steps):
    pred_y = mlp.predict(np.array(stack[-N:]).reshape(1,-1))
    pred_y=pred_y*0.88
    res.append(pred_y[0])
    stack.append(pred_y[0])
res=pd.DataFrame(res,columns=['capacity_GW'])
res['date']=date_range

df_concat=pd.concat([df[['capacity_GW','date']],res])
# 格式化日期为年-月的字符串格式
df_concat['year_month'] = df_concat['date'].dt.strftime('%Y-%m')
# 筛选出每年12月份的数据
december_data = df_concat[df_concat['date'].dt.month == 12]
# 提取年份和相应的数据列
result = december_data.groupby(december_data['date'].dt.year)['capacity_GW'].sum().reset_index()
result.columns = ['date', 'capacity_GW']

# 根据日期范围绘制折线图
plt.figure(figsize=(20, 6))

# 划分2015到2024年的数据
data_2015_to_2024 = result[(result['date']>= 2015) & (result['date']<= 2024)]
plt.plot(data_2015_to_2024['date'].apply(str), data_2015_to_2024['capacity_GW'], label='2015-2024', color='blue')

# 划分2024到2035年的数据
data_2024_to_2035 = result[(result['date']>= 2024) & (result['date']<= 2035)]
plt.plot(data_2024_to_2035['date'].apply(str), data_2024_to_2035['capacity_GW'], label='2024-2035', color='red')

plt.xlabel('date')
plt.ylabel('capacity_GW')
plt.title('zhejaing_solar_capacity_GW from 2015 to 2035')
plt.legend()
plt.show()

# 定义cdf函数
def calculate_P(t, a=30, b=5.3759):
    k = -(t / a)**b
    return 1 - np.exp(k)

scrap_amount = [] ##报废量
years=[i for i in range(2025,2036)]
for year in years:
    xx=result[(result['date']< year)] ##找到year之前所有年份
    cap=result[(result['date']==year)]['capacity_GW'].iloc[0] ##找到当年的容量
    xx['life']=year-xx['date']
    xx['increase']=cap-xx['capacity_GW']
    xx['scrap_prob']=xx['life'].apply(calculate_P)
    scrap=sum(xx['increase']*xx['scrap_prob'])
    scrap_amount.append(scrap)

final_res=pd.DataFrame()
final_res['year']=years
final_res['total_scrap_amount']=scrap_amount

renkou={
    'hangzhou': 18.8*0.01,
    'ningbo': 14.6*0.01,
    'wenzhou': 14.7*0.01,
    'jiaxing': 8.4*0.01,
    'huzhou': 5.2*0.01,
    'shaoxing': 8.1*0.01,
    'jinhua': 10.8*0.01,
    'quzhou': 3.5*0.01,
    'zhoushan': 1.8*0.01,
    'taizhou': 10.2*0.01,
    'lishui': 3.8*0.01
}

for city in renkou.keys():
    final_res[f'{city}_']=final_res['total_scrap_amount']*renkou[city]
print(final_res)