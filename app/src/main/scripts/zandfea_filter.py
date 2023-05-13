import csv
import numpy as np
import matplotlib.pyplot as plt

def read_csv(filename):
    data = []
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            data.append(float(row[0]))
    return data

# 读取存储的 x 和 feature 数据
# feature_data = np.array( read_csv('feature.csv'))*-32 +1.2
feature_data = np.array(read_csv('feature.csv'))
z_data = np.array(read_csv('z.csv')) * 0.07


# 一阶互补滤波参数
alpha = 0.1

filtered_feature_data = [feature_data[0]]  # 初始化滤波后的数据
filtered_z_data = [z_data[0]]
# 应用一阶互补滤波器
for i in range(1, len(feature_data)):
    filtered_feature = alpha * feature_data[i] + (1 - alpha) * filtered_feature_data[i-1]
    filtered_feature_data.append(filtered_feature)

for i in range(1, len(z_data)):
    filtered_z = alpha * z_data[i] + (1 - alpha) * filtered_z_data[i-1]
    filtered_z_data.append(filtered_z)

zf_data =np.array(filtered_z_data) - np.array(filtered_feature_data)
# 绘制原始数据和滤波后的数据
plt.plot(z_data, label='z_data')
plt.plot(feature_data, label='feature_data')
plt.plot(zf_data, label='zf_data')
plt.xlabel('time')
plt.ylabel('value')
plt.legend()
plt.show()
