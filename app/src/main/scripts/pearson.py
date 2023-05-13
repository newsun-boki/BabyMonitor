import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

def read_csv(filename):
    data = []
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            data.append(float(row[0]))
    return data

# 读取存储的 x 和 feature 数据
x_data = read_csv('x.csv')
y_data = read_csv('y.csv')
z_data = read_csv('z.csv')
acc_data = read_csv('acc.csv')
feature_data = read_csv('feature.csv')

# 计算皮尔逊相关系数
corr_x = pearsonr(feature_data, x_data)[0]
corr_y = pearsonr(feature_data, y_data)[0]
corr_z = pearsonr(feature_data, z_data)[0]
corr_acc = pearsonr(feature_data, acc_data)[0]

# 设置柱形颜色
colors = ['r', 'g', 'b', 'y']

# 绘制柱形图
labels = ['x', 'y', 'z', 'acc']
correlations = [corr_x, corr_y, corr_z, corr_acc]

plt.bar(labels, correlations)
plt.xlabel('Variable')
plt.ylabel('Pearson Correlation')
plt.title('Correlation between feature and variables')

# 添加数值标签
for i, val in enumerate(correlations):
    plt.text(i, val, f'{val:.2f}', ha='center', va='bottom')

plt.show()
