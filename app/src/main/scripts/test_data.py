import matplotlib.pyplot as plt 
import numpy as np
from scipy import signal

# 读取数据
data = np.genfromtxt("2023-05-05 14_40_imu_data.csv", delimiter=',')
# 将字符串类型转换为短整型类型
data = data.astype(np.int16)
data = np.resize(data[:,0], (787200, ))

# 定义短时傅里叶变换的参数
# 定义短时傅里叶变换的参数
window = signal.windows.hamming(1024)
nperseg = 1024
noverlap = 512

# 计算短时傅里叶变换
f, t, Sxx = signal.spectrogram(data, 48000)
print(t)

# 绘制时域图
ax1 = plt.figure(1)
plt.plot(np.arange(data.shape[0]) / 48000,data)
plt.title('Time Domain Signal')
plt.ylabel('Sensor Value')

# 绘制频谱图
ax2 = plt.figure(2)
plt.pcolormesh(t, f*2.0, np.log10(Sxx))
plt.title('STFT Magnitude')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')

plt.show()
