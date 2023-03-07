import csv
import pandas as pd
import matplotlib.pyplot as plt


# 获取每行数据
data = []
with open("./output/2023-03-07_21-47-30_circle/detect_datas.csv", "r") as f:
    csv_reader = csv.reader(f)
    for row in csv_reader:
        data.append([float(i) for i in row])

# 绘制每行数据
for i in range(len(data)):
    fig, ax = plt.subplots()
    ax.plot(data[i])
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy of Line {}".format(i+1))
    plt.show()
