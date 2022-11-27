package com.boyu.babymonitor;


import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;

import android.Manifest;
import android.annotation.SuppressLint;
import android.content.Context;
import android.content.pm.PackageManager;
import android.graphics.Color;
import android.media.AudioFormat;
import android.media.AudioManager;
import android.media.AudioRecord;
import android.media.AudioTrack;
import android.media.MediaRecorder;
import android.os.Bundle;
import android.view.View;
import android.widget.CompoundButton;
import android.widget.EditText;
import android.widget.ImageView;
import android.widget.Switch;
import android.widget.TextView;
import android.widget.Toast;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.LinkedList;
import java.util.List;
import java.util.Queue;

import fft.Complex;
import fft.FFT;
import util.WaveUtil;
import view.WaveShowView;

public class MainActivity extends AppCompatActivity {

    //画图相关
    private WaveUtil waveUtil;

    //界面相关
    private TextView textViewStatus;
    private EditText editTextGainFactor;
    private Switch switchButton;
    private ImageView starImage;

    //音频相关
    private AudioRecord audioRecord;
    private AudioTrack audioTrack;
    private AudioManager audioManager;

    //超声波数据
    private double[] sinData;
    private int ultrasonicFrequency = 20500;

    private int intBufferSize;
    private short[] shortPlayAudioData;
    private short[] shortRecordAudioData;
    //信号处理
    Queue<Integer> maxFrequencyIndexs;
    float [][]frameOut;//分窗后的结果
    float [][]filteredFrameOut;//滤波后结果
    double handVelocity;
    List<Double> handVelocitys;
    private double amplitude;
    private double[] highPassFilterNumerator;
    private double[] highPassFilterDenominator;

    private int intGain;
    private boolean isActive = false;

    private Thread thread;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.RECORD_AUDIO}, PackageManager.PERMISSION_GRANTED);
        audioManager = (AudioManager) this.getSystemService(Context.AUDIO_SERVICE);
        //界面控件查找
        textViewStatus = findViewById(R.id.textViewStatus);
        editTextGainFactor = findViewById(R.id.editTextGainFactor);
        switchButton = findViewById(R.id.switch1);
        starImage = findViewById(R.id.imageView);
        switchButton.setOnCheckedChangeListener(new CompoundButton.OnCheckedChangeListener() {
            @Override
            public void onCheckedChanged(CompoundButton compoundButton, boolean isChecked) {
                if(isChecked){
                    onSetListener(true);
                }else{
                    onSetListener(false);
                }
            }
        });
        maxFrequencyIndexs = new LinkedList<>();
        handVelocitys = new ArrayList<>();
        highPassFilterInitial();
        //画图相关
        waveUtil = new WaveUtil();

    }

    //使用matlab生成的6阶，采样率48000的高通巴特沃斯滤波器
    void highPassFilterInitial(){
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
    //切换扬声器
    private void onSetListener(boolean isCall){
        if(audioManager != null){
            if(isCall){
                openSpeaker();
            }else{
                closeSpeaker();
            }
        }

    }

    private void closeSpeaker() {
        if(audioManager.isSpeakerphoneOn()){
            audioManager.setSpeakerphoneOn(false);
//            audioManager.setRouting(AudioManager.MODE_NORMAL, AudioManager.ROUTE_EARPIECE, AudioManager.ROUTE_ALL);
            audioManager.setMode(AudioManager.MODE_IN_COMMUNICATION);
//            setVolumeControlStream(AudioManager.STREAM_VOICE_CALL);
            Toast.makeText(this,"Speaker off",Toast.LENGTH_SHORT).show();
        }
    }

    private void openSpeaker() {
        if(!audioManager.isSpeakerphoneOn()){
            audioManager.setSpeakerphoneOn(true);
            audioManager.setMode(AudioManager.MODE_NORMAL);

            Toast.makeText(this,"Speaker on",Toast.LENGTH_SHORT).show();
        }
    }

    public void buttonStart(View view){
        isActive = true;
        intGain = Integer.parseInt(editTextGainFactor.getText().toString());
        textViewStatus.setText("Active");

        thread = new Thread(new Runnable() {
            @Override
            public void run() {
                threadLoop();
            }
        });
        thread.start();
        startPlot(view);
    }

    public void buttonStop(View view){
        isActive = false;
        audioTrack.stop();
        audioRecord.stop();
        textViewStatus.setText("Stopped");
        stopPlot(view);
    }

    @SuppressLint("MissingPermission")
    private void threadLoop(){
        int intRecordSampleRate = AudioTrack.getNativeOutputSampleRate(AudioManager.STREAM_MUSIC);
        intRecordSampleRate = 48000;
        intBufferSize = AudioRecord.getMinBufferSize(intRecordSampleRate, AudioFormat.CHANNEL_IN_MONO,AudioFormat.ENCODING_PCM_16BIT);
        shortRecordAudioData = new short[intBufferSize];
        audioRecord = new AudioRecord(MediaRecorder.AudioSource.MIC,intRecordSampleRate,AudioFormat.CHANNEL_IN_STEREO,
                AudioFormat.ENCODING_PCM_16BIT,intBufferSize);
        audioTrack = new AudioTrack(AudioManager.STREAM_MUSIC,intRecordSampleRate,AudioFormat.CHANNEL_IN_STEREO,
                AudioFormat.ENCODING_PCM_16BIT,intBufferSize,AudioTrack.MODE_STREAM);

        audioTrack.setPlaybackRate(intRecordSampleRate);
    //设置超声波数据
        setSinData(32000F,ultrasonicFrequency / 2, 0.3,intRecordSampleRate);//频率似乎被放大了两倍
        shortPlayAudioData = new short[(int)(0.2 * intRecordSampleRate)];
        for (int i = 0; i < shortPlayAudioData.length; i++) {
            shortPlayAudioData[i] = (short) sinData[i];
        }
        
        audioRecord.startRecording();
        audioTrack.play();

        while (isActive){
            for (int i = 0; i < shortPlayAudioData.length; i++) {
                shortPlayAudioData[i] = (short)Math.min((shortPlayAudioData[i] * intGain),Short.MAX_VALUE);
//                waveUtil.setFloatData((float)(shortRecordAudioData[i]));
                 

            }

            int writeResult = audioTrack.write(shortPlayAudioData,0,shortPlayAudioData.length);
            if(writeResult >= 0){
                //success
            }else{
                continue;
            }
            //接收回声存储在shortRecordAudioData
            audioRecord.read(shortRecordAudioData,0,shortRecordAudioData.length);
            dataPreprocess(intRecordSampleRate);
            for (int i = 0; i < shortRecordAudioData.length; i++) {
//                shortRecordAudioData[i] = (short)Math.min((shortRecordAudioData[i] * intGain),Short.MAX_VALUE);
//                waveUtil.setFloatData((float)(shortRecordAudioData[i]));
            }


        }
    }

    //开始绘制波形
    public void startPlot(View view) {
        WaveShowView waveShowView = findViewById(R.id.waveview);
        waveUtil.showWaveData(waveShowView);
    }
    //停止绘制波形
    public void stopPlot(View view) {
        waveUtil.stop();
    }

    //生成超声波数据
    private void setSinData(double volume,int frequency,double duration,int sampleRate){
        //音量1000, 频率20000,持续事件1s,采样率48000
        int sinDataLength = (int)(duration * sampleRate);
        sinData = new double[sinDataLength];
        for(double i = 0, j = 0; i < duration && j < sinDataLength; i += 1.0/(double) sampleRate, j++){
            sinData[(int)j] = (double)(volume * Math.sin(2 * Math.PI * frequency * i));
        }
    }

    private void dataPreprocess(int sampleRate){
        //初始化一些数据
        int dataLength = shortRecordAudioData.length;//3584
        double frameShiftTime = 0.01;//帧移时长/s
        double windowLengthTime = 0.02;//窗口时长/s 这个越大频率分辨率越高
        int frameShift = (int) (sampleRate * frameShiftTime);//帧移长度
        int windowLength =(int) (sampleRate * windowLengthTime);//窗口长度
        int frameNumber = (dataLength - windowLength)/frameShift + 1;
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

        //高通滤波
        for (int i = 0; i < frameNumber; i++) {
            for(int j = 0; j < windowLength; j++){
                if(j < highPassFilterNumerator.length - 1){
                    filteredFrameOut[i][j] = frameOut[i][j];
                }else {
                    filteredFrameOut[i][j] = (float)highPassFilterNumerator[0]*frameOut[i][j];
                    for (int k = 1; k < highPassFilterNumerator.length; k++) {
                        filteredFrameOut[i][j] = (float)(highPassFilterNumerator[k] * frameOut[i][j - k]
                                            - highPassFilterDenominator[k]*filteredFrameOut[i][j - k]);
                    }
                }

            }
        }
        double averageHandVelocity = 0;
        double averageAmplitude = 0;
        for (int i = 0; i < frameNumber; i++) {
            getHandVelocity(i, windowLength, sampleRate);//获取手的速度

            averageHandVelocity += Math.abs(handVelocity) / (double)frameNumber;
            averageAmplitude += amplitude / (double)frameNumber;

        }
        //通过一个标志来显示是否检测到手
        if(averageAmplitude < 0.03){
            starImage.setBackgroundColor(Color.RED);
        }else{
            if(averageHandVelocity < 1) {
                starImage.setBackgroundColor(Color.GREEN);
            }else{
                starImage.setBackgroundColor(Color.YELLOW);
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
        waveUtil.setFloatData((float)(mid));


    }

    //输入为第i帧，窗长，采样率
    void getHandVelocity(int i, int windowLength, int sampleRate){
        int N = (int)(Math.pow(2,Math.floor(Math.log(windowLength)/Math.log(2))));
        Double[] x;
        Double[] x1 = new Double[N];

        //傅里叶变换计算
        Complex[] input = new Complex[N];//声明复数数组
        for (int j = 0; j <= N-1; j++) {
            input[j] = new Complex(filteredFrameOut[i][j], 0);}//将实数数据转换为复数数据
        input = FFT.getFFT(input, N);//傅里叶变换
        x=Complex.toModArray(input);//计算傅里叶变换得到的复数数组的模值
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

        double frequencyDifference = ultrasonicFrequency - maxFrequency;
        handVelocity = frequencyDifference / (ultrasonicFrequency + maxFrequency) * 340.29;
        if(x1Max < 0.16){//经过实测得到一般至少幅值大于0.1的才有效
            handVelocity = 0;
            x1Max = 0.0;
        }
        amplitude = x1Max;
        System.out.println(String.valueOf(maxFrequency) + " " + String.valueOf(handVelocity) + " " + String.valueOf(x1Max));
    }
}