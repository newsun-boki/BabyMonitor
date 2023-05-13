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
x_data = read_csv('x.csv')
feature_data = np.array( read_csv('feature.csv'))
y_data = np.array( read_csv('y.csv'))
z_data = np.array( read_csv('z.csv'))
acc_data = np.array( read_csv('acc.csv'))
# 平滑处理
# x_smooth = np.linspace(min(np.arange(len(x_data))), max(np.arange(len(x_data))), 300)  # 生成更多的点用于平滑
# x_spline = make_interp_spline(np.arange(len(x_data)), x_data)
# x_smooth = x_spline(x_smooth)

# feature_spline = make_interp_spline(np.arange(len(x_data)), feature_data)
# feature_smooth = feature_spline(x_smooth)

# 绘制原始数据和平滑的曲线
plt.plot(x_data,label='x')
plt.plot(y_data,label='y')
plt.plot(z_data,label='z')
plt.plot(acc_data,label='acc')
plt.plot(feature_data,label='fea')
plt.xlabel('time')
plt.ylabel('value')
plt.legend()
plt.show()
