import os
import numpy as np
import scipy.io.wavfile as wav
from scipy.fftpack import fft
from python_speech_features import mfcc
from sklearn.metrics.pairwise import cosine_similarity
import pyaudio
import wave
import matplotlib.pyplot as plt

def load_wav(file):
    '''
    读取一个wav文件，返回声音信号的时域谱矩阵、播放时间、通道数和帧数
    '''
    load_wav_file = wave.open(file, "rb")  # 開啟wav檔
    load_num_frames = load_wav_file.getnframes()  # 得到音框
    load_num_channels = load_wav_file.getnchannels()  # 得到聲道数
    load_framerate = load_wav_file.getframerate()  # 音框赫茲數
    load_num_sample_width = load_wav_file.getsampwidth()  # 得到實際的bit寬度，即每一幀率的字节数
    
    str_data = load_wav_file.readframes(load_num_frames)  # 讀取全部的幀
    load_wav_file.close()  # 關閉
    load_wave_data = np.frombuffer(str_data, dtype=np.int16)  # 将声音文件数据转换为数组矩阵形式
    load_wave_data = load_wave_data.reshape(-1, load_num_channels)  # 按照声道数将数组整形，单声道时候是一列数组，双声道时候是两列的矩阵
    return load_wave_data, load_framerate, load_num_channels, load_num_frames

def record_audio(filename):
    """录制与标准语音相同长度的用户输入的语音"""
    load_wave_data, load_framerate, load_num_channels, load_num_frames = load_wav(filename)  # 将文件名传递给 load_wav 函数
    record_audio_frames = load_num_frames  # 設定音框
    record_FORMAT = pyaudio.paInt16
    record_CHANNELS = load_num_channels #通道數
    record_RATE = load_framerate # 音檔赫茲數

    p = pyaudio.PyAudio()
    stream = p.open(format=record_FORMAT, channels=record_CHANNELS, rate=record_RATE, input=True, frames_per_buffer=record_audio_frames)
    print("Start recording...")

    frames = []
    seconds = 4  # 录制4秒，可以根据需求调整

    for i in range(0, int(record_RATE / record_audio_frames * seconds)):
        data = stream.read(record_audio_frames)
        frames.append(data)
    print("Recording stopped")

    stream.stop_stream()
    stream.close()

    wf = wave.open(filename, 'wb')
    wf.setnchannels(record_CHANNELS)
    wf.setsampwidth(p.get_sample_size(record_FORMAT))
    wf.setframerate(record_RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

def endpoint_detection(data_signal, frame_size, overlap, framerate):  # 添加帧率参数
    """端点检测"""
    energy = np.sum(data_signal ** 2, axis=frame_size)  # 计算每个音框的能量
    threshold = np.mean(energy) * 1.5  # 设置能量阈值
    endpoints = np.where(energy > threshold)[0]  # 找到超过阈值的帧索引
    start = endpoints[0] - int(frame_size / 2)  # 起始点为第一个超过阈值的帧的前一半帧
    end = endpoints[-1] + int(frame_size / 2)  # 终点为最后一个超过阈值的帧的后一半帧
    return start, end

def calculate_volume_curve(signal, frame_size, overlap, framerate):  # 添加帧率参数
    """计算音量强度曲线"""
    print("Signal shape:", signal.shape)  # 添加调试语句
    start, end = endpoint_detection(signal, frame_size, overlap, framerate)  # 传递帧率参数
    frames = signal[start:end]

def plot_volume_curve(signal, rate, frame_size, overlap):
    """绘制音量强度曲线"""
    mag_curve = calculate_volume_curve(signal, frame_size, overlap, rate)  # 传递帧率参数
    time = np.arange(len(mag_curve)) * (frame_size * (1 - overlap)) / rate
    plt.plot(time, mag_curve)
    plt.xlabel('Time (s)')
    plt.ylabel('Average Magnitude')
    plt.title('Volume Intensity Curve')
    plt.show()

def main():
    # 标准语音文件路径
    wav_filename = r"C:\Users\USER\Desktop\A.wav"
    
    # 读取标准语音文件
    wave_data, framerate, num_channels, num_frames = load_wav(wav_filename)
    
    # 打印通道数
    print("Number of channels:\n", num_channels)
    print("Number of wave_data:\n", wave_data)
    print("Number of framerate:\n", framerate)
    print("Number of num_frames:\n", num_frames)
    # 设置音框大小和重叠参数
    frame_size = 512
    overlap = 170
    plot_volume_curve(wave_data, framerate, frame_size, overlap)

if __name__ == "__main__":
    main()