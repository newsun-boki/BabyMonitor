import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

def read_csv(filename):
    data = []
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            data.append(float(row[0]))
    return data

# 读取存储的 x 和 feature 数据
# feature_data = np.array( read_csv('feature.csv'))*-32 +1.2
feature_data = np.array( read_csv('feature.csv'))
z_data =np.array( read_csv('z.csv'))*0.07
# 平滑处理
# x_smooth = np.linspace(min(np.arange(len(z_data))), max(np.arange(len(z_data))), 300)  # 生成更多的点用于平滑
# x_spline = make_interp_spline(np.arange(len(z_data)), z_data)
# x_smooth = x_spline(x_smooth)

# feature_spline = make_interp_spline(np.arange(len(feature_data)), feature_data)
# feature_smooth = feature_spline(x_smooth)

# 绘制原始数据和平滑的曲线
# plt.plot(x_smooth,label='z_smooth')
# plt.plot(feature_smooth,label='feature_smooth')
plt.plot(z_data,label='z_data')
plt.plot(feature_data,label='feature_data')
plt.xlabel('time')
plt.ylabel('value')
plt.legend()
plt.show()
