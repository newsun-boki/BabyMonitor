package com.boyu.babymonitor;

import android.annotation.SuppressLint;
import android.os.Handler;
import android.os.Message;
import android.util.Log;
import android.widget.Toast;

import java.net.DatagramPacket;
import java.net.DatagramSocket;
import java.net.InetAddress;

public class Socket {
    MainActivity mainActivity;
    private InetAddress address;
    private int port;
    DatagramSocket socket;

    private final int HANDLER_MSG_TELL_RECV = 0x124;
    @SuppressLint("HandlerLeak")
    Handler handler01 = new Handler(){
        public void handleMessage(Message msg,MainActivity mainActivity){
            //接受到服务器信息时执行
            Toast.makeText(mainActivity,(msg.obj).toString(),Toast.LENGTH_LONG).show();
        }
    };

    public Socket(String str_address,int port, MainActivity mainActivity) {
        try {
            this.address = InetAddress.getByName(str_address);
            this.port = port;
            this.mainActivity = mainActivity;
            socket = new DatagramSocket();
            Log.i("socket","创建socket成功");
        }
        catch (Exception e){
            e.printStackTrace();
            Log.e("socket","创建socket失败");
        }
    }

    protected void finalize(){
        socket.close();
        Log.i("socket","socket关闭");
    }
    public void sendMessage(String message){
        byte[] data = message.getBytes();
        DatagramPacket packet = new DatagramPacket(data, data.length, address, port);
        try {
            socket.send(packet);
            Log.i("socket","socket发送成功:" + message);
        }catch (Exception e){
            e.printStackTrace();
            Log.e("socket","socket发送失败");
        }
    }

    //测试用
    public void startUdpThread() {
        new Thread() {
            @Override
            public void run() {
                try {
                    /*
                     * 向服务器端发送数据
                     */
                    // 1.定义服务器的地址、端口号、数据
                    InetAddress address = InetAddress.getByName("192.168.43.125");
                    int port = 555;
                    byte[] data = "用户名：admin;密码：123".getBytes();
                    // 2.创建数据报，包含发送的数据信息
                    DatagramPacket packet = new DatagramPacket(data, data.length, address, port);
                    // 3.创建DatagramSocket对象
                    DatagramSocket socket = new DatagramSocket();
                    // 4.向服务器端发送数据报
                    int i = 0;
                    while(i < 5){
                        System.out.println("send 用户名：admin;密码：123 ");
                        Log.i("socket","send message");
                        socket.send(packet);
//                        socket.receive(packet2);
//                        String reply = new String(data2, 0, packet2.getLength());
//                        Log.i("udp","我是客户端，服务器说：" + reply);
                    }

                    /*
                     * 接收服务器端响应的数据
                     */
                    // 1.创建数据报，用于接收服务器端响应的数据
                    byte[] data2 = new byte[1024];
                    DatagramPacket packet2 = new DatagramPacket(data2, data2.length);
                    // 2.接收服务器响应的数据
//                    socket.receive(packet2);
//                    // 3.读取数据
//                    String reply = new String(data2, 0, packet2.getLength());
//
//                    System.out.println("我是客户端，服务器说：" + reply);
//                    Log.i("udp","我是客户端，服务器说：" + reply);
//                    Message msg = handler01.obtainMessage(HANDLER_MSG_TELL_RECV, reply);
//                    msg.sendToTarget();

                    // 4.关闭资源
                    socket.close();

                }
                catch (Exception e) {
                }
            }
        }.start();
    }


}
