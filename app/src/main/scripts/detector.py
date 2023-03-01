from multiprocessing import Process,Queue
import keyboard
import torch
import torch.nn as nn
import csv
import time
import os
from datetime import datetime
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

        

def detect(detect_data:Queue,detect_imu:Queue):
    detect_datas = []
    detect_accs = []
    now = datetime.now()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
    "cpu")
    net = ConvNet().to(device)
    mode = 'collect'
    before = time.time()
    if mode == 'collect': 
        # 获取当前时间并格式化为字符串
        current_time = time.strftime("%Y-%m-%d_%H-%M-%S")

        # 创建一个以当前时间命名的文件夹
        folder_name = f"./output/{current_time}"
        os.mkdir(folder_name)
        file_acc = open('{}/detect_accs.csv'.format(folder_name), mode='w', newline='')
        file_data = open('{}/detect_datas.csv'.format(folder_name), mode='w', newline='')
        try:
            while True:
                detect_datas.append(detect_data.get())
                detect_accs.append(detect_imu.get())
                if len(detect_datas) > 15:
                    detect_datas.pop(0)
                if len(detect_accs)>15:
                    detect_accs.pop(0)
                if keyboard.is_pressed('s'):
                    pressed = time.time()
                    if pressed - before > 0.3: 
                        writer_acc = csv.writer(file_acc)
                        writer_acc.writerow(detect_accs)
                        writer_data = csv.writer(file_data)
                        writer_data.writerow(detect_datas)
                        print("save:")
                        print(detect_accs)
                        before =time.time()

                if keyboard.is_pressed('q'):
                    break
        finally:
            file_acc.close()
            file_data.close()
    else:
        while True:
                detect_datas.append(detect_data.get())
                detect_accs.append(detect_imu.get())
                if len(detect_datas) > 15:
                    detect_datas.pop(0)
                if len(detect_accs)>15:
                    detect_accs.pop(0)

                if keyboard.is_pressed('q'):
                    break

if __name__ == "__main__":
    detect() 