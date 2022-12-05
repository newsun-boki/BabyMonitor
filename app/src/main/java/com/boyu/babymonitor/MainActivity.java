package com.boyu.babymonitor;


import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;

import android.Manifest;
import android.annotation.SuppressLint;
import android.content.Context;
import android.content.pm.PackageManager;
import android.graphics.Color;
import android.hardware.Sensor;
import android.hardware.SensorManager;
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
    private int intRecordSampleRate = 48000;

    //超声波数据
    private double[] sinData;
    private int ultrasonicFrequency = 20500;

    private int intBufferSize;
    private short[] shortPlayAudioData;
    private short[] shortRecordAudioData;

    SignalProcessor signalProcessor;    //信号处理

    //imu
    private SensorManager sensorManager;
    private Sensor sensor;


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
        signalProcessor = new SignalProcessor(intRecordSampleRate,ultrasonicFrequency);
        //画图相关
        waveUtil = new WaveUtil();

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
//        if(audioManager.isSpeakerphoneOn()){
            audioManager.setSpeakerphoneOn(true);
//            audioManager.setRouting(AudioManager.MODE_NORMAL, AudioManager.ROUTE_EARPIECE, AudioManager.ROUTE_ALL);Pa
            audioManager.setMode(AudioManager.MODE_IN_COMMUNICATION);
//            setVolumeControlStream(AudioManager.STREAM_VOICE_CALL);
            Toast.makeText(this,"Speaker off",Toast.LENGTH_SHORT).show();
//        }
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

    public void setStarImage(int detectionStatus){
        switch (detectionStatus){
            case SignalProcessor.NO_ACTIVITY:
                starImage.setBackgroundColor(Color.RED);
                break;
            case SignalProcessor.WRONG_ACTIVITY:
                starImage.setBackgroundColor(Color.YELLOW);
                break;
            case SignalProcessor.DETECTED_ACTIVITY:
                starImage.setBackgroundColor(Color.GREEN);
                break;
        }
    }

    @SuppressLint("MissingPermission")
    private void threadLoop(){
//        int intRecordSampleRate = AudioTrack.getNativeOutputSampleRate(AudioManager.STREAM_MUSIC);
        intBufferSize = AudioRecord.getMinBufferSize(intRecordSampleRate, AudioFormat.CHANNEL_IN_MONO,AudioFormat.ENCODING_PCM_16BIT);
        shortRecordAudioData = new short[intBufferSize];
        audioRecord = new AudioRecord(MediaRecorder.AudioSource.MIC,intRecordSampleRate,AudioFormat.CHANNEL_IN_STEREO,
                AudioFormat.ENCODING_PCM_16BIT,intBufferSize);
        audioTrack = new AudioTrack(AudioManager.STREAM_MUSIC,intRecordSampleRate,AudioFormat.CHANNEL_IN_STEREO,
                AudioFormat.ENCODING_PCM_16BIT,intBufferSize,AudioTrack.MODE_STREAM);

        audioTrack.setPlaybackRate(intRecordSampleRate);
    //设置超声波数据
        setSinData(30000F,ultrasonicFrequency / 2, 0.1,intRecordSampleRate);//频率似乎被放大了两倍
        shortPlayAudioData = new short[(int)(0.1 * intRecordSampleRate)];
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
            //
            signalProcessor.setShortRecordAudioData(shortRecordAudioData);
            signalProcessor.dataProcess();
//            dataPreprocess(intRecordSampleRate);
//            waveUtil.setFloatData((float) signalProcessor.getAverageAmplitude());
            waveUtil.setFloatData((float) signalProcessor.getDifferencePhase());
            setStarImage(signalProcessor.getDetectionStatus());

            //条用

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

}