import csv
import numpy as np
from scipy.spatial.distance import cosine

def read_csv(filename):
    data = []
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            data.append(float(row[0]))
    return data

# 读取存储的 x 和 feature 数据
x_data = read_csv('acc.csv')
feature_data = read_csv('feature.csv')

# 将数据转换为 NumPy 数组
x_data = np.array(x_data)
feature_data = np.array(feature_data)

# 计算余弦距离
distance = cosine(x_data, feature_data)

print(f"Cosine distance: {distance}")
