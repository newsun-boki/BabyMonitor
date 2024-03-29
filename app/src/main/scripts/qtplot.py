__author__ = 'Ted'

from pyqtgraph import PlotWidget
from PyQt5 import QtCore,QtWidgets
import numpy as np
import pyqtgraph as pq
class Window(QtWidgets.QWidget):
    def __init__(self,qv,data_type):
        super().__init__()
        # 设置下尺寸
        self.resize(600,600)
        # 添加 PlotWidget 控件
        self.plotWidget_ted = PlotWidget(self)
        if data_type == 'acc':
            self.setWindowTitle("acc")
            self.plotWidget_ted.setRange(yRange=(-1, 1))
        else:
            self.setWindowTitle("feature")
            self.plotWidget_ted.setRange(yRange=(-0.1, 0.1))

        # 设置该控件尺寸和相对位置
        self.plotWidget_ted.setGeometry(QtCore.QRect(25,25,550,550))

        # 仿写 mode1 代码中的数据
        # 生成 300 个正态分布的随机数
        self.data1 = np.random.normal(size=100)
        self.qv = qv
        self.vs = []
        self.curve1 = self.plotWidget_ted.plot(self.data1, name="mode1")

        # 设定定时器
        self.timer = pq.QtCore.QTimer()
        # 定时器信号绑定 update_data 函数
        # self.timer.timeout.connect(self.show_audio)
        self.timer.timeout.connect(self.update_data)
        # 定时器间隔50ms，可以理解为 50ms 刷新一次数据
        self.timer.start(10)

    def update_data(self):
        if self.qv.empty() == False:
            v = self.qv.get()
            self.vs.append(v)
            if len(self.vs) > 50:
                del(self.vs[0])
        # 数据填充到绘制曲线中
            self.curve1.setData(self.vs)

    def keyPressEvent(self, event):
        # 按下'q'键关闭窗口
        if event.key() == QtCore.Qt.Key_Q:
            self.close()

def qtplot(qv, type):
    import sys
    # PyQt5 程序固定写法
    app = QtWidgets.QApplication(sys.argv)

    # 将绑定了绘图控件的窗口实例化并展示
    window = Window(qv,type)
    window.show()

    # PyQt5 程序固定写法
    sys.exit(app.exec())

if __name__ == '__main__':
    import sys
    # PyQt5 程序固定写法
    app = QtWidgets.QApplication(sys.argv)

    # 将绑定了绘图控件的窗口实例化并展示
    window = Window()
    window.show()

    # PyQt5 程序固定写法
    sys.exit(app.exec())
    