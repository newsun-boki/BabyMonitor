from multiprocessing import Process,Queue
import keyboard
import torch
import torch.nn as nn
import csv
import time
import os
from datetime import datetime
import matplotlib.pyplot as plt
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.fc1 = nn.Linear(16*50, 10)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(-1, 16*50)
        x = self.fc1(x)
        return x

        

def detect(detect_data:Queue,detect_imu_x:Queue,detect_imu_y:Queue,detect_imu_z:Queue,):
    detect_datas = []
    detect_imu_xs = []
    detect_imu_ys = []
    detect_imu_zs = []
    now = datetime.now()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
    "cpu")
    net = ConvNet().to(device)
    mode = 'collect'
    before = time.time()
    cnt = 0
    if mode == 'collect': 
        # 获取当前时间并格式化为字符串
        current_time = time.strftime("%Y-%m-%d_%H-%M-%S")
        data_lens = 30
        first_time_save = True
        try:
            while True:
                detect_datas.append(detect_data.get())
                detect_imu_xs.append(detect_imu_x.get())
                detect_imu_ys.append(detect_imu_y.get())
                detect_imu_zs.append(detect_imu_z.get())
                if len(detect_datas) > data_lens:
                    detect_datas.pop(0)
                if len(detect_imu_xs)>data_lens:
                    detect_imu_xs.pop(0)
                if len(detect_imu_ys)>data_lens:
                    detect_imu_ys.pop(0)
                if len(detect_imu_zs)>data_lens:
                    detect_imu_zs.pop(0)
                if keyboard.is_pressed('s'):
                    pressed = time.time()
                    if pressed - before > 0.3:
                        if first_time_save == True:
                            first_time_save = False
                            # 创建一个以当前时间命名的文件夹
                            folder_name = f"./output/{current_time}"
                            os.mkdir(folder_name)
                            file_x = open('{}/detect_imu_xs.csv'.format(folder_name), mode='w', newline='')
                            file_y = open('{}/detect_imu_ys.csv'.format(folder_name), mode='w', newline='')
                            file_z = open('{}/detect_imu_zs.csv'.format(folder_name), mode='w', newline='')
                            file_data = open('{}/detect_datas.csv'.format(folder_name), mode='w', newline='') 
                        else:
                            plt.close()

                        writer_x = csv.writer(file_x)
                        writer_x.writerow(detect_imu_xs)
                        writer_y = csv.writer(file_y)
                        writer_y.writerow(detect_imu_ys)
                        writer_z = csv.writer(file_z)
                        writer_z.writerow(detect_imu_zs)
                        writer_data = csv.writer(file_data)
                        writer_data.writerow(detect_datas)
                        print("save:" + str(cnt))
                        cnt = cnt + 1
                        #画图
                        plt.plot(detect_datas)
                        plt.xlabel("value")
                        plt.ylabel("num")
                        plt.ylim((-0.1,0.1))
                        plt.show()
                        before =time.time()

                if keyboard.is_pressed('q'):
                    break
        finally:
            file_x.close()
            file_y.close()
            file_z.close()
            file_data.close()
            if os.path.getsize('{}/detect_datas.csv'.format(folder_name)) == 0:
                os.remove(folder_name)
    # else:
    #     while True:
    #             detect_datas.append(detect_data.get())
    #             detect_accs.append(detect_imu.get())
    #             if len(detect_datas) > 15:
    #                 detect_datas.pop(0)
    #             if len(detect_accs)>15:
    #                 detect_accs.pop(0)

    #             if keyboard.is_pressed('q'):
    #                 break

if __name__ == "__main__":
    detect() 