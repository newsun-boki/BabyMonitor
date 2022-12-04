package com.boyu.babymonitor;

import android.graphics.Color;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedList;
import java.util.List;
import java.util.Queue;

import fft.Complex;
import fft.FFT;

public class SignalProcessor {


    private short[] shortRecordAudioData;
    public void setShortRecordAudioData(short[] shortRecordAudioData) {
        this.shortRecordAudioData = shortRecordAudioData;
    }

    private int sampleRate;

    //信号处理
    float [][]frameOut;//分窗后的结果
    float [][]filteredFrameOut;//滤波后结果
    double handVelocity;
    List<Double> handVelocitys;
    private double amplitude;
    private double phase;
    private double lastAmplitude = 0;
    //滤波器
    private double[] highPassFilterNumerator;
    private double[] highPassFilterDenominator;

    //超声波频率
    private int ultrasonicFrequency;

    private double averageAmplitude;
    private double averageHandVelocity;

    public double getAveragePhase() {
        return averagePhase;
    }
    public double getDifferencePhase() {
        return differencePhase;
    }
    //  相位
    private double averagePhase;
    private double lastPhase;
    private double differencePhase;




    public double getAverageAmplitude() {
        return averageAmplitude;
    }
    //检测状态
    private int detectionStatus;
    public static final int NO_ACTIVITY = 0;
    public static final int WRONG_ACTIVITY = 1;
    public static final int DETECTED_ACTIVITY = 2;


    public SignalProcessor(){};

    public SignalProcessor(int sampleRate, int ultrasonicFrequency){
        this.sampleRate = sampleRate;
        this.ultrasonicFrequency = ultrasonicFrequency;
        handVelocitys = new ArrayList<>();
        highPassFilterInitial();
    }

    public int getDetectionStatus() {
        return detectionStatus;
    }

    public void dataProcess(){
        //初始化一些数据
        int dataLength = shortRecordAudioData.length;//3584
        double frameShiftTime = 0.02;//帧移时长/s
        double windowLengthTime = 0.02;//窗口时长/s 这个越大频率分辨率越高
        int frameShift = (int) (sampleRate * frameShiftTime);//帧移长度
        int windowLength =(int) (sampleRate * windowLengthTime);//窗口长度
        int frameNumber = (dataLength - windowLength)/frameShift + 1;

        initFrameOut(frameNumber,windowLength,frameShift);//初始化数据存储的frameout

        highPassFilter(frameNumber,windowLength);//高通滤波

        //取所有帧的均值
        averageHandVelocity = 0;
        averageAmplitude = 0;
        averagePhase = 0;
        for (int i = 0; i < frameNumber; i++) {
            getHandVelocity(i, windowLength, sampleRate);//获取手的速度
            averageHandVelocity += Math.abs(handVelocity) / (double)frameNumber;
            averageAmplitude += amplitude / (double)frameNumber;
            averagePhase += Math.abs(phase)/ (double)frameNumber;
        }
        //通过一个标志来显示是否检测到手
        if(averageAmplitude < 0.03){
            detectionStatus = NO_ACTIVITY;
        }else{
            if(averageHandVelocity < 1) {
                detectionStatus = DETECTED_ACTIVITY;
            }else{
                averageAmplitude = 0.0;
                phase=0.0;
                detectionStatus = WRONG_ACTIVITY;
            }
        }
        //中值滤波
        handVelocitys.add(averageHandVelocity);
        if(handVelocitys.size() > 3){
            handVelocitys.remove(0);
        }
        List<Double> sortedList = new ArrayList<Double>(handVelocitys);
        Collections.sort(sortedList);
        double mid = sortedList.get(sortedList.size() / 2);
        lastAmplitude = averageAmplitude;
        differencePhase = averagePhase - lastPhase;
        lastPhase = averagePhase;

        System.out.println(" averageAmplitude:" + averageAmplitude + " differencePhase:" + differencePhase + " frameNumber:"+frameNumber);

    }

    private void initFrameOut(int frameNumber, int windowLength, int frameShift){
        frameOut = new float[frameNumber][];
        for (int i = 0; i < frameNumber; i++) {
            frameOut[i] = new float[windowLength];
            for(int j = 0; j < windowLength; j++){
                frameOut[i][j] = (float)(shortRecordAudioData[i * frameShift + j]);
            }
        }
        filteredFrameOut = new float[frameNumber][];
        for (int i = 0; i < frameNumber; i++) {
            filteredFrameOut[i] = new float[windowLength];

        }
    }

    //使用matlab生成的6阶，采样率48000的高通巴特沃斯滤波器
    private void highPassFilterInitial(){
        highPassFilterNumerator = new double[7];
        highPassFilterNumerator[0] =  0.007585107580853602850246009126067292527;
        highPassFilterNumerator[1] = -0.045510645485121646591775146362124360166;
        highPassFilterNumerator[2] =  0.113776613712804230971187280374579131603;
        highPassFilterNumerator[3] = -0.151702151617072400480168425929150544107;
        highPassFilterNumerator[4] =  0.113776613712804536282519052292627748102;
        highPassFilterNumerator[5] = -0.045510645485121868636380071393432444893;
        highPassFilterNumerator[6] =  0.007585107580853650555141598488262388855;

        highPassFilterDenominator = new double[7];
        highPassFilterDenominator[0] = 1;
        highPassFilterDenominator[1] = 1.485051528776588636304722967906855046749;
        highPassFilterDenominator[2] = 1.603614295818286628048099373700097203255;
        highPassFilterDenominator[3] = 0.924060108900980559099025413161143660545;
        highPassFilterDenominator[4] = 0.359233258743223093922836142155574634671;
        highPassFilterDenominator[5] = 0.075611177960372949469203263106464873999;
        highPassFilterDenominator[6] = 0.007322146251062889091287821941023139516;

    }
    private void highPassFilter(int frameNumber, int windowLength){
        //高通滤波
        for (int i = 0; i < frameNumber; i++) {
            for(int j = 0; j < windowLength; j++){
                if(j < highPassFilterNumerator.length - 1){
                    filteredFrameOut[i][j] = frameOut[i][j];
                }else {
//                    filteredFrameOut[i][j] = (float)highPassFilterNumerator[0]*frameOut[i][j];
//                    for (int k = 1; k < highPassFilterNumerator.length; k++) {
//                        filteredFrameOut[i][j] += (float)(highPassFilterNumerator[k] * frameOut[i][j - k]
//                                            - highPassFilterDenominator[k]*filteredFrameOut[i][j - k]);
//                    }
                    double a =   0.240151945772565655889962954461225308478 * frameOut[i][j]
                            + -0.720455837317697023181040094641502946615 * frameOut[i][j-1]
                            +  0.720455837317697023181040094641502946615 * frameOut[i][j-2]
                            + -0.240151945772565655889962954461225308478 * frameOut[i][j-3];
                    double b = -0.48070916354772053047383906232425943017  * filteredFrameOut[i][j-1]
                            +  0.394605575830941690540498711925465613604 * filteredFrameOut[i][j-2]
                            + -0.045900826801862991410896341903935535811 * filteredFrameOut[i][j-3];
                    filteredFrameOut[i][j] = (float) (a - b);
                }

            }
        }
    }

    //输入为第i帧，窗长，采样率
    private void getHandVelocity(int i, int windowLength, int sampleRate){
        int N = (int)(Math.pow(2,Math.floor(Math.log(windowLength)/Math.log(2))));
        Double[] x,p;
        Double[] x1 = new Double[N];

        //傅里叶变换计算
        Complex[] input = new Complex[N];//声明复数数组
        for (int j = 0; j <= N-1; j++) {
            input[j] = new Complex(filteredFrameOut[i][j], 0);}//将实数数据转换为复数数据
        input = FFT.getFFT(input, N);//傅里叶变换
        x=Complex.toModArray(input);//计算傅里叶变换得到的复数数组的模值
        p=Complex.toPhaseArray(input);
        for(int j=0;j<=N-1;j++) {
            //的模值数组除以N再乘以2
            x1[j]=x[j]/N*2;

        }
        //得到幅指最大频率
        double frequencyInterval = sampleRate / N;//频率间隔
        int topFrequency = ultrasonicFrequency +2000;
        int bottomFrequency = ultrasonicFrequency -500;
        Double x1Max = -1.0;
        int x1MaxIndex = -1;
        for(int j=0;j<=N/2;j++) {
            if(j * frequencyInterval * 2 < bottomFrequency){
                continue;
            }
            if(x1[j] > x1Max){
                x1Max = x1[j];
                x1MaxIndex = j;
            }
            if(j * frequencyInterval * 2 > topFrequency){//大于21kHZ之后就不取了
                break;
            }

        }

        //最终频率间隔为44100/N
        double maxFrequency = x1MaxIndex * frequencyInterval * 2;
        double frequencyDifference = 20274 - maxFrequency;
        handVelocity = frequencyDifference / (20274 + maxFrequency) * 340.29;
        if(x1Max < 0.16){//经过实测得到一般至少幅值大于0.1的才有效
            handVelocity = 0;
            x1Max = 0.0;
            p[x1MaxIndex]=0.0;
        }
        amplitude = x1Max;
//        System.out.println(String.valueOf(maxFrequency) + " " + String.valueOf(frequencyDifference) + " " + String.valueOf(x1Max)+ " "+p[x1MaxIndex]);
        phase = Math.abs(p[x1MaxIndex]);
    }
}
