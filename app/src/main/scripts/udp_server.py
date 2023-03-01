import socket
import keyboard


class UDPServer:
    def __init__(self, address='', port=555):
        self.address = address
        self.port = port
        self.socket = None
        
    def create_socket(self):
        self.socket = socket.socket(type=socket.SOCK_DGRAM)
        self.socket.bind((self.address, self.port))
        print(f'Server listening on {self.address}:{self.port}...')
        
    def receive_socket(self):
        
        # print("begin receive")
        msg, addr = self.socket.recvfrom(1024)
        # print("receive " + msg.decode('utf-8') + " from " + str(addr))
        # print("发送")
        # self.socket.sendto('hello'.encode('utf-8'), addr)
        return msg.decode('utf-8')
    
    def close_socket(self):
        self.socket.close()
        print("Connection closed.")


if __name__ == "__main__":
    udpServer = UDPServer('',555)
    udpServer.create_socket()

    while True:
        msg = udpServer.receive_socket()
        print(float(msg))
        if keyboard.is_pressed('q'):
            break
    udpServer.close_socket()
        