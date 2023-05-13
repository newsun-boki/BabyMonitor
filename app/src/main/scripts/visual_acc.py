import csv
import pandas as pd
import matplotlib.pyplot as plt

def filter(data,alpha):
    filtered_data =  [data[0]]
    for i in range(1, len(data)):
        filtered = alpha * data[i] + (1 - alpha) * filtered_data[i-1]
        filtered_data.append(filtered)  
    return filtered_data
root_dir = "./output/2023-03-23_22-47-14_ZUpDown"
# 获取每行数据
data_x = []
with open(root_dir +"/detect_imu_xs.csv", "r") as f:
    csv_reader = csv.reader(f)
    for row in csv_reader:
        data_x.append([float(i) for i in row])
data_y = []
with open(root_dir+"/detect_imu_ys.csv", "r") as f:
    csv_reader = csv.reader(f)
    for row in csv_reader:
        data_y.append([float(i) for i in row])
data_z = []
with open(root_dir+"/detect_imu_zs.csv", "r") as f:
    csv_reader = csv.reader(f)
    for row in csv_reader:
        data_z.append([float(i) for i in row])

# 绘制每行数据
for i in range(len(data_x)):
    data_x[i] = filter(data_x[i],0.2)
    data_y[i] = filter(data_y[i],0.2)
    data_z[i] = filter(data_z[i],0.2)
    plt.plot(data_x[i],label='x')
    plt.plot(data_y[i],label='y')
    plt.plot(data_z[i],label = 'z')
    plt.xlabel("time")
    plt.ylabel("value")
    plt.legend()
    plt.show()