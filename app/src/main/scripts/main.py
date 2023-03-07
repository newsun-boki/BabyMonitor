from udp_server import UDPServer
import keyboard
from detector import detect
from multiprocessing import Process,Queue
import math

visualize_data = 'acc' #acc or feature



def msgProcessor(msg:str):
    values = msg.split(',')
    feature = round(float(values[0]),5)
    x = round(float(values[1]),5)
    y = round(float(values[2]),5)
    z = round(float(values[3]),5)
    return feature,x,y,z


def main():
    plot_datas_feature = Queue()
    plot_datas_imu = Queue()
    detect_data = Queue()
    detect_imu_x = Queue()
    detect_imu_y = Queue()
    detect_imu_z = Queue()
    from qtplot import qtplot
    plot_feature = Process(target=qtplot,args=(plot_datas_feature,'feature',))
    plot_imu = Process(target=qtplot,args=(plot_datas_imu,'acc',))
    detector = Process(target=detect, args=(detect_data,detect_imu_x,detect_imu_y,detect_imu_z,))
    plot_feature.start()
    plot_imu.start()
    detector.start()
    udpServer = UDPServer('',555)
    udpServer.create_socket()
    datas = []
    while True:
        #接受消息
        msg = udpServer.receive_socket()
        feature,x,y,z= msgProcessor(msg)
        acc= round(math.sqrt(x*x + y*y + z*z),5)
        #收集用于检测的数据
        detect_data.put(feature)
        if detect_data.qsize()>2:
            detect_data.get()
        detect_imu_x.put(x)
        if detect_imu_x.qsize()>2:
            detect_imu_x.get()
        detect_imu_y.put(y)
        if detect_imu_y.qsize()>2:
            detect_imu_y.get()
        detect_imu_z.put(z)
        if detect_imu_z.qsize()>2:
            detect_imu_z.get()

        #绘图
        plot_datas_imu.put(z)
        if plot_datas_imu.qsize() > 2:
            plot_datas_imu.get()
        plot_datas_feature.put(feature-z/15.0)
        if plot_datas_feature.qsize() > 2:
            plot_datas_feature.get()

        datas.append(msg)
        if len(datas) > 20:
            datas.pop(0)
        if keyboard.is_pressed('q'):
            break
    udpServer.close_socket()


if __name__ == "__main__":
    main()
