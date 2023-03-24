import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import time
import os
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        
        out, _ = self.lstm(x, (h0, c0))
        
        out = self.fc(out[:, -1, :])
        
        return out

# Hyperparameters
input_size = 4
hidden_size = 64
num_layers = 2
num_classes = 4 #增加类的话改这里
batch_size = 8
learning_rate = 0.001
num_epochs = 300
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Instantiate the model
model = LSTM(input_size, hidden_size,num_layers, num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

rootpath_circle = "./output/2023-03-07_23-13-47_circle/"
detect_datas_circle=np.loadtxt(rootpath_circle + "detect_datas.csv",delimiter=",")
detect_imu_zs_circle=np.loadtxt(rootpath_circle + "detect_imu_zs.csv",delimiter=",")
detect_imu_ys_circle=np.loadtxt(rootpath_circle + "detect_imu_ys.csv",delimiter=",")
detect_imu_xs_circle=np.loadtxt(rootpath_circle + "detect_imu_xs.csv",delimiter=",")
label_circle = np.array([0 for _ in range(detect_datas_circle.shape[0])])

rootpath_updown = "./output/2023-03-08_01-17-35_UpDown/"
detect_datas_updown=np.loadtxt(rootpath_updown+"detect_datas.csv",delimiter=",")
detect_imu_zs_updown=np.loadtxt(rootpath_updown+"detect_imu_zs.csv",delimiter=",")
detect_imu_ys_updown=np.loadtxt(rootpath_updown+"detect_imu_ys.csv",delimiter=",")
detect_imu_xs_updown=np.loadtxt(rootpath_updown+"detect_imu_xs.csv",delimiter=",")
label_updown = np.array([1 for _ in range(detect_datas_updown.shape[0])])

rootpath_leftright = "./output/2023-03-23_22-11-18_RightLeft/"
detect_datas_leftright=np.loadtxt(rootpath_leftright+"detect_datas.csv",delimiter=",")
detect_imu_zs_leftright=np.loadtxt(rootpath_leftright+"detect_imu_zs.csv",delimiter=",")
detect_imu_ys_leftright=np.loadtxt(rootpath_leftright+"detect_imu_ys.csv",delimiter=",")
detect_imu_xs_leftright=np.loadtxt(rootpath_leftright+"detect_imu_xs.csv",delimiter=",")
label_leftright = np.array([2 for _ in range(detect_datas_leftright.shape[0])])

rootpath_zupdown = "./output/2023-03-23_22-47-14_ZUpDown/"
detect_datas_zupdown=np.loadtxt(rootpath_zupdown+"detect_datas.csv",delimiter=",")
detect_imu_zs_zupdown=np.loadtxt(rootpath_zupdown+"detect_imu_zs.csv",delimiter=",")
detect_imu_ys_zupdown=np.loadtxt(rootpath_zupdown+"detect_imu_ys.csv",delimiter=",")
detect_imu_xs_zupdown=np.loadtxt(rootpath_zupdown+"detect_imu_xs.csv",delimiter=",")
label_zupdown = np.array([3 for _ in range(detect_datas_zupdown.shape[0])])

detect_datas = np.concatenate((detect_datas_circle,detect_datas_updown,detect_datas_leftright,detect_datas_zupdown),axis=0)
detect_imu_zs = np.concatenate((detect_imu_zs_circle,detect_imu_zs_updown,detect_imu_zs_leftright,detect_imu_zs_zupdown),axis=0)
detect_imu_ys = np.concatenate((detect_imu_ys_circle,detect_imu_ys_updown,detect_imu_ys_leftright,detect_imu_ys_zupdown),axis=0)
detect_imu_xs = np.concatenate((detect_imu_xs_circle,detect_imu_xs_updown,detect_imu_xs_leftright,detect_imu_xs_zupdown),axis=0)

# Merge detect_datas and detect_imu_zs into one input sequence
data = np.concatenate([detect_datas[:, :, np.newaxis],detect_imu_zs[:, :, np.newaxis],detect_imu_ys[:, :, np.newaxis],detect_imu_xs[:, :, np.newaxis]],axis=2)
labels = np.concatenate((label_circle,label_updown,label_leftright,label_zupdown))
print(data.shape)
print(labels)
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

x_train = torch.from_numpy(x_train).float()
y_train = torch.from_numpy(y_train).long()
x_test = torch.from_numpy(x_test).float()
y_test = torch.from_numpy(y_test).long()

# 将数据转换为三维张量
x_train = x_train.view(-1, 30, 4)
x_test = x_test.view(-1, 30, 4)
print(x_train.shape)
# 将数据封装为数据集对象
train_dataset = TensorDataset(x_train, y_train)
test_dataset = TensorDataset(x_test, y_test)
# 将数据集对象封装为DataLoader对象
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

def test_accuracy():
    # 测试模型
    with torch.no_grad():
        correct = 0
        total = 0
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            outputs = model(batch_x)
            _, predicted = torch.max(outputs.data, 1)
            
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum()
            
        # print(f'Test Accuracy: {100 * correct / total}%')
        return 100 * correct / total
# 训练模型
best_accuracy = -1
best_state_dict = None
for epoch in range(num_epochs):
    for batch_x, batch_y in train_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        
        outputs = model(batch_x)

        loss = criterion(outputs, batch_y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    if (epoch+1) % 5 == 0:
        accuracy = test_accuracy()
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_state_dict = model.state_dict()
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Accuracy:%{accuracy},Best Accuracy:%{best_accuracy}')
current_time = time.strftime("%Y-%m-%d_%H-%M-%S")
folder_name = f"./weights/{current_time}"
os.mkdir(folder_name)
torch.save(best_state_dict,f'{folder_name}/best.pt')