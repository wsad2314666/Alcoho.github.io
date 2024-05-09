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
    读取一个wav文件，返回声音信号的时域谱矩阵、播放时间和通道数
    '''
    wav_file = wave.open(file, "rb")  # 打开一个wav格式的声音文件流
    num_frames = wav_file.getnframes()  # 获取帧数
    num_channels = wav_file.getnchannels()  # 获取声道数
    framerate = wav_file.getframerate()  # 获取帧速率
    num_sample_width = wav_file.getsampwidth()  # 获取实例的比特宽度，即每一帧的字节数
    
    str_data = wav_file.readframes(num_frames)  # 读取全部的帧
    wav_file.close()  # 关闭流
    wave_data = np.frombuffer(str_data, dtype=np.int16)  # 将声音文件数据转换为数组矩阵形式
    wave_data = wave_data.reshape(-1, num_channels)  # 按照声道数将数组整形，单声道时候是一列数组，双声道时候是两列的矩阵
    return wave_data, framerate, num_channels

def record_audio(filename):
    """錄製與標準語音相同長度的使用者輸入的語音"""
    CHUNK = 4096  # 進一步增加帧大小
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100

    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    print("Start recording...")

    frames = []
    seconds = 4  # 錄製3秒，可以根據需求調整

    for i in range(0, int(RATE / CHUNK * seconds)):
        data = stream.read(CHUNK)
        frames.append(data)
    print("Recording stopped")

    stream.stop_stream()
    stream.close()

    wf = wave.open(filename, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
    
def endpoint_detection(signal, frame_size, overlap):
    """端点检测"""
    energy = np.sum(signal ** 2, axis=1)  # 计算每个音框的能量
    threshold = np.mean(energy) * 1.5  # 设置能量阈值
    endpoints = np.where(energy > threshold)[0]  # 找到超过阈值的帧索引
    start = endpoints[0] - int(frame_size / 2)  # 起始点为第一个超过阈值的帧的前一半帧
    end = endpoints[-1] + int(frame_size / 2)  # 终点为最后一个超过阈值的帧的后一半帧
    return start, end

def calculate_volume_curve(signal, frame_size, overlap):
    """计算音量强度曲线"""
    print("Signal shape:", signal.shape)  # 添加调试语句
    start, end = endpoint_detection(signal, frame_size, overlap)
    frames = []
    for i in range(start, end - frame_size, int(frame_size * (1 - overlap))):
        frames.append(signal[i:i + frame_size])
    frames = np.array(frames)
    
    # 处理当frames的长度为1时的情况
    if len(frames.shape) == 1:
        mag_curve = np.abs(frames)
    else:
        mag_curve = np.mean(np.abs(frames), axis=1)  # 计算每个音框的幅度
    
    return mag_curve

def plot_volume_curve(signal, rate, frame_size, overlap):
    """绘制音量强度曲线"""
    mag_curve = calculate_volume_curve(signal, frame_size, overlap)
    time = np.arange(len(mag_curve)) * (frame_size * (1 - overlap)) / rate
    plt.plot(time, mag_curve)
    plt.xlabel('Time (s)')
    plt.ylabel('Average Magnitude')
    plt.title('Volume Intensity Curve')
    plt.show()

def main():
    # 標準語音檔案路徑
    wav_filename = r"C:\Users\USER\Desktop\A.wav"
    
    # 讀取標準語音檔案
    wave_data, fs, num_channels = load_wav(wav_filename)
    
    # 打印通道数
    print("Number of channels:", num_channels)
    
    # 設定音框大小和重疊參數
    frame_size = 512
    overlap = 170
    
    # 繪製音量強度曲線
    plot_volume_curve(wave_data[:, :num_channels], fs, frame_size, overlap)  # 使用正确维度的音频数据

if __name__ == "__main__":
    main()