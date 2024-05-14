import os
import numpy as np
import scipy.io.wavfile as wav
from scipy.fftpack import fft
from python_speech_features import mfcc
from sklearn.metrics.pairwise import cosine_similarity
import pyaudio
import wave
import matplotlib.pyplot as plt
import librosa

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



def endpoint_detection(data_signal, frame_size, overlap, framerate):  
    """端点检测"""
    energy = np.sum(data_signal ** 2, axis=1)  # 沿着第二个维度求和
    threshold = np.mean(energy) * 1.5  # 设置能量阈值
    endpoints = np.where(energy > threshold)[0]  # 找到超过阈值的帧索引
    start = endpoints[0] - int(frame_size / 2)  # 起始点为第一个超过阈值的帧的前一半帧
    end = endpoints[-1] + int(frame_size / 2)  # 终点为最后一个超过阈值的帧的后一半帧
    return start, end

def calculate_volume_curve(signal, frame_size, overlap, framerate):  
    """计算音量强度曲线"""
    print("Signal shape:", signal.shape)  # 添加调试语句
    start, end = endpoint_detection(signal, frame_size, overlap, framerate)  # 传递帧率参数
    frames = signal[start:end]
    # 在这里计算音量强度曲线，假设计算结果存储在 mag_curve 变量中
    # 计算能量
    energy = librosa.feature.rms(y=signal, frame_length=frame_size, hop_length=frame_size - overlap)[0]
    mag_curve = energy  # 这里填入计算音量强度曲线的代码
    return mag_curve

# def plot_volume_curve(signal, rate, frame_size, overlap):
#     """绘制音量强度曲线"""
#     mag_curve = calculate_volume_curve(signal, frame_size, overlap, rate)  # 传递帧率参数
#     time = np.arange(len(mag_curve)) * (frame_size * (1 - overlap)) / rate
#     plt.plot(time, mag_curve)
#     plt.xlabel('Time (s)')
#     plt.ylabel('Average Magnitude')
#     plt.title('Volume Intensity Curve')
#     plt.show()

# def plot_volume_curve(signal, framerate, frame_size, overlap):
#     """绘制音量强度曲线"""
#     time = librosa.frames_to_time(range(len(signal)), sr=framerate, hop_length=frame_size - overlap)
#     mag_curve = calculate_volume_curve(signal, framerate, frame_size, overlap)
#     plt.plot(time, mag_curve)
#     plt.xlabel('Time (s)')
#     plt.ylabel('Average Magnitude')
#     plt.title('Volume Intensity Curve')
#     plt.show()

def calculate_mfcc(data, rate):
    """計算MFCC特徵"""
    return mfcc(data, rate)

def similarity(input_features, reference_features):
    """計算輸入語音與參考語音之間的相似度"""
    # 如果輸入特徵是三維的，將其展平為二維
    if input_features.ndim == 3:
        input_features = input_features.reshape(input_features.shape[0], -1)
    return cosine_similarity(input_features, reference_features)

def main():
    # 錄製語音
    user_input_filename = "user_input.wav"
    record_audio(user_input_filename)

    # 讀取使用者輸入的語音檔案並計算MFCC特徵
    user_wave_data, user_framerate, user_num_channels, user_num_frames = record_audio(user_input_filename)
    user_mfcc = calculate_mfcc(user_wave_data, user_framerate)

    # 標準語音路徑
    wav_filename = 'D:/wsad2314666/Alcoho.github.io/語音評分系統/A.wav'
    # 讀取標準語音檔案並計算MFCC特徵
    load_wave_data, load_framerate, load_num_channels, load_num_frames = load_wav(wav_filename)
    load_mfcc = calculate_mfcc(load_wave_data, load_framerate)
    # 打印通道数
    print("Number of channels:\n", load_num_channels)
    print("Number of wave_data:\n", load_wave_data)
    print("Number of framerate:\n", load_framerate)
    print("Number of num_frames:\n", load_num_frames)
    # 計算使用者輸入語音與標準語音的相似度
    sim_scores = similarity(load_mfcc, user_mfcc)
    print("使用者輸入語音與標準語音的相似度比對結果：", sim_scores)
    # 设置音框大小和重叠参数
    frame_size = 512
    overlap = 170
    #plot_volume_curve(wave_data, framerate, frame_size, overlap)
    #plot_volume_curve(wave_data[:, 0], framerate, frame_size, overlap)
if __name__ == "__main__":
    main()