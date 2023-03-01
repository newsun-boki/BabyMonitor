from udp_server import UDPServer
import keyboard
from qtplot import qtplot
from detector import detect
from multiprocessing import Process,Queue
import math

def msgProcessor(msg:str):
    values = msg.split(',')
    feature = round(float(values[0]),5)
    x = round(float(values[1]),5)
    y = round(float(values[2]),5)
    z = round(float(values[3]),5)
    return feature,x,y,z


def main():
    plot_datas = Queue()
    detect_data = Queue()
    detect_imu = Queue()
    plot = Process(target=qtplot,args=(plot_datas,))
    detector = Process(target=detect, args=(detect_data,detect_imu))
    plot.start()
    detector.start()
    udpServer = UDPServer('',555)
    udpServer.create_socket()
    datas = []
    while True:
        #接受消息
        msg = udpServer.receive_socket()
        feature,x,y,z= msgProcessor(msg)
        acc= round(math.sqrt(x*x + y*y + z*z),5)

        detect_data.put(feature)
        if detect_data.qsize()>2:
            detect_data.get()
        detect_imu.put(acc)
        if detect_imu.qsize()>2:
            detect_imu.get()

        #绘图
        plot_datas.put(feature)
        if plot_datas.qsize() > 2:
            plot_datas.get()
        datas.append(msg)
        if len(datas) > 20:
            datas.pop(0)
        if keyboard.is_pressed('q'):
            break
    udpServer.close_socket()


if __name__ == "__main__":
    main()
