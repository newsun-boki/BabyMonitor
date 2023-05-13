from multiprocessing import Process,Queue
import keyboard
import torch
import torch.nn as nn
import csv
import time
import os
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        device = torch.device("cpu")
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        
        out, _ = self.lstm(x, (h0, c0))
        
        out = self.fc(out[:, -1, :])
        
        return out
def detect(detect_data:Queue,detect_imu_x:Queue,detect_imu_y:Queue,detect_imu_z:Queue,):
    detect_datas = []
    detect_imu_xs = []
    detect_imu_ys = []
    detect_imu_zs = []
    #载入模型
    input_size = 4
    hidden_size = 64
    num_layers = 2
    num_classes = 4
    device = torch.device("cpu")
    model = LSTM(input_size, hidden_size,num_layers, num_classes).to(device)
    model.load_state_dict(torch.load('./weights/2023-05-13_12-01-23/best.pt'))
    gestures = ['circle','no_gesture','shaking','yupdown']
    mode = 'dcollect'
    before = time.time()
    cnt = 0
    data_lens = 30
    
    if mode == 'collect': 
        # 获取当前时间并格式化为字符串
        current_time = time.strftime("%Y-%m-%d_%H-%M-%S")
        first_time_save = True
        collect_times = 0
        begin_collect = False
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
                if begin_collect:
                    collect_times = collect_times + 1
                    if collect_times > data_lens:
                        begin_collect = False
                        collect_times = 0
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
                if keyboard.is_pressed('s'):
                    pressed = time.time()
                    begin_collect = True
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
                        
                        #画图
                        # plt.plot(detect_datas)
                        # plt.xlabel("value")
                        # plt.ylabel("num")
                        # plt.ylim((-0.1,0.1))
                        # plt.show()
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
    else:
        begin_detect = False
        detect_times = 0
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
            if begin_detect:
                detect_times = detect_times + 1
                if detect_times> data_lens:
                    detect_times = 0
                    begin_detect = False
                    detect_datas_array = np.array(detect_datas)
                    detect_imu_zs_array = np.array(detect_imu_zs)
                    detect_imu_ys_array = np.array(detect_imu_ys)
                    detect_imu_xs_array = np.array(detect_imu_xs)
                    data = np.concatenate([detect_datas_array[:, np.newaxis],detect_imu_zs_array[:, np.newaxis],detect_imu_ys_array[:, np.newaxis],detect_imu_xs_array[:, np.newaxis]],axis=1)
                    data_tensor = torch.from_numpy(data).float()
                    data_tensor =data_tensor.view(-1, 30, 4)
                    with torch.no_grad():
                        data_tensor = data_tensor.to(device)
                        outputs = model(data_tensor)
                        _, predicted = torch.max(outputs.data, 1)
                        print(gestures[int(predicted)])
            if keyboard.is_pressed('d'):
                pressed = time.time()
                begin_detect =True
                if pressed - before > 0.3:
                   
                        
                    before =time.time()
            if keyboard.is_pressed('q'):
                    break

if __name__ == "__main__":
    detect() 