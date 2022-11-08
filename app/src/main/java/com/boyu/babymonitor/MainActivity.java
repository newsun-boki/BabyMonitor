package com.boyu.babymonitor;

import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;

import android.Manifest;
import android.annotation.SuppressLint;
import android.content.Context;
import android.content.pm.PackageManager;
import android.media.AudioFormat;
import android.media.AudioManager;
import android.media.AudioRecord;
import android.media.AudioTrack;
import android.media.MediaRecorder;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.CompoundButton;
import android.widget.EditText;
import android.widget.Switch;
import android.widget.TextView;
import android.widget.Toast;

import util.WaveUtil;
import view.WaveShowView;

public class MainActivity extends AppCompatActivity {

    //画图相关
    private WaveUtil waveUtil;

    //界面相关
    private TextView textViewStatus;
    private EditText editTextGainFactor;
    private Switch switchButton;

    //音频相关
    private AudioRecord audioRecord;
    private AudioTrack audioTrack;
    private AudioManager audioManager;

    private int intBufferSize;
    private short[] shortAudioData;

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
        intBufferSize = AudioRecord.getMinBufferSize(intRecordSampleRate, AudioFormat.CHANNEL_IN_MONO,AudioFormat.ENCODING_PCM_16BIT);
        shortAudioData = new short[intBufferSize];
        audioRecord = new AudioRecord(MediaRecorder.AudioSource.MIC,intRecordSampleRate,AudioFormat.CHANNEL_IN_STEREO,
                AudioFormat.ENCODING_PCM_16BIT,intBufferSize);
        audioTrack = new AudioTrack(AudioManager.STREAM_MUSIC,intRecordSampleRate,AudioFormat.CHANNEL_IN_STEREO,
                AudioFormat.ENCODING_PCM_16BIT,intBufferSize,AudioTrack.MODE_STREAM);

        audioTrack.setPlaybackRate(intRecordSampleRate);

        audioRecord.startRecording();
        audioTrack.play();

        while (isActive){
            audioRecord.read(shortAudioData,0,shortAudioData.length);

            for (int i = 0; i < shortAudioData.length; i++) {
                shortAudioData[i] = (short)Math.min((shortAudioData[i] * intGain),Short.MAX_VALUE);
                waveUtil.setFloatData((float)(shortAudioData[i]));
            }
            System.out.println(shortAudioData[0]);
            audioTrack.write(shortAudioData,0,shortAudioData.length);
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
}