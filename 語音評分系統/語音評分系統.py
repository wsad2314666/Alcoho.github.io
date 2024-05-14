from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import librosa
import sounddevice as sd
from scipy.ndimage import maximum_filter1d
import matplotlib.pyplot as plt
import os

app = Flask(__name__)

# 載入音檔
def load_audio(file_path):
    audio, sr = librosa.load(file_path, sr=None)
    return audio, sr

# 提取 MFCC 特徵
def extract_mfcc(audio, sr):
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    return mfccs

# 正規化 MFCC 特徵
def normalize_mfcc(mfccs):
    return (mfccs - np.mean(mfccs)) / np.std(mfccs)

# 計算相似度分數
def compute_similarity_score(mfccs_A, mfccs_B):
    return np.mean(np.abs(mfccs_A - mfccs_B)) * 100

# 去除靜音部分
def remove_silence(audio, threshold=0.02):
    non_silent_intervals = librosa.effects.split(audio, top_db=threshold)
    non_silent_audio = np.concatenate([audio[start:end] for start, end in non_silent_intervals])
    return non_silent_audio

# 預處理雜音
def preprocess_audio(audio):
    # 雜音預處理，例如濾波等
    # 這裡我們使用 maximum_filter1d 函數來濾波
    filtered_audio = maximum_filter1d(audio, size=3)
    return filtered_audio

# 主函式
def main(audio_file_A,audio_file_B):
    # 載入音檔 A
    audio_A, sr_A = load_audio(r"D:\wsad2314666\Alcoho.github.io\語音評分系統\A.wav")
    # 載入音檔 B
    audio_B, sr_B = load_audio(r"D:\wsad2314666\Alcoho.github.io\語音評分系統\output.wav")

    # 去除靜音部分
    audio_A = remove_silence(audio_A)
    audio_B = remove_silence(audio_B)

    # 預處理雜音
    audio_preA = preprocess_audio(audio_A)
    audio_preB = preprocess_audio(audio_B)

    # 提取 MFCC 特徵
    mfccs_A = extract_mfcc(audio_preA, sr_A)
    mfccs_B = extract_mfcc(audio_preB, sr_A)

    # 使兩個音檔的 MFCC 特徵具有相同的維度
    min_length = min(mfccs_A.shape[1], mfccs_B.shape[1])
    mfccs_A = mfccs_A[:, :min_length]
    mfccs_B = mfccs_B[:, :min_length]

    # 正規化 MFCC 特徵
    mfccs_A_normalized = normalize_mfcc(mfccs_A)
    mfccs_B_normalized = normalize_mfcc(mfccs_B)

    # 計算相似度分數
    score = 100-compute_similarity_score(mfccs_A_normalized, mfccs_B_normalized)

    return score
if __name__ == "__main__":
    audio_file_A = (r"D:\wsad2314666\Alcoho.github.io\語音評分系統\A.wav")
    audio_file_B = (r"D:\wsad2314666\Alcoho.github.io\語音評分系統\output.wav")
    similarity_score = main(audio_file_A,audio_file_B)
    print("相似度分數:", similarity_score)
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # 載入音檔 A
        audio_A, sr_A = load_audio('./A.wav')
        # 載入音檔 B
        audio_B, sr_B = load_audio('./output.wav')
        # 去除靜音部分
        audio_A = remove_silence(audio_A)
        audio_B = remove_silence(audio_B)
        # 預處理雜音
        audio_preA = preprocess_audio(audio_A)
        audio_preB = preprocess_audio(audio_B)
        # 提取 MFCC 特徵
        mfccs_A = extract_mfcc(audio_A, sr_A)
        mfccs_B = extract_mfcc(audio_B, sr_A)
        # 使兩個音檔的 MFCC 特徵具有相同的維度
        min_length = min(mfccs_A.shape[1], mfccs_B.shape[1])
        mfccs_A = mfccs_A[:, :min_length]
        mfccs_B = mfccs_B[:, :min_length]
        # 正規化 MFCC 特徵
        mfccs_A_normalized = normalize_mfcc(mfccs_A)
        mfccs_B_normalized = normalize_mfcc(mfccs_B)
        # 計算相似度分數
        score1 = 100-compute_similarity_score(mfccs_A_normalized, mfccs_B_normalized)
        score =score1(int)
        return render_template('index.html', score=score)
    return render_template('index.html')

if __name__ == '__main__':
    app.debug = True
    app.run()
# import os
# import numpy as np
# import scipy.io.wavfile as wav
# from scipy.fftpack import fft
# from python_speech_features import mfcc
# from sklearn.metrics.pairwise import cosine_similarity
# import pyaudio
# import wave
# import matplotlib.pyplot as plt

# def load_wav(file):
#     """讀取.wav檔案並回傳取樣率和音訊資料"""
#     rate, data = wav.read(file)
#     return rate, data

# def record_audio(filename):
#     """錄製與標準語音相同長度的使用者輸入的語音"""
#     CHUNK = 4096  # 進一步增加帧大小
#     FORMAT = pyaudio.paInt16
#     CHANNELS = 1
#     RATE = 44100

#     p = pyaudio.PyAudio()
#     stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
#     print("Start recording...")

#     frames = []
#     seconds = 4  # 錄製3秒，可以根據需求調整

#     for i in range(0, int(RATE / CHUNK * seconds)):
#         data = stream.read(CHUNK)
#         frames.append(data)
#     print("Recording stopped")

#     stream.stop_stream()
#     stream.close()

#     wf = wave.open(filename, 'wb')
#     wf.setnchannels(CHANNELS)
#     wf.setsampwidth(p.get_sample_size(FORMAT))
#     wf.setframerate(RATE)
#     wf.writeframes(b''.join(frames))
#     wf.close()

# def calculate_mfcc(data, rate):
#     """計算MFCC特徵"""
#     return mfcc(data, rate)

# def load_reference_data(wav_filename):
#     """載入參考語音數據"""
#     # 讀取參考語音檔案
#     rate_ref, data_ref = load_wav(wav_filename)

#     # 計算參考語音的MFCC特徵
#     reference_mfcc = calculate_mfcc(data_ref, rate_ref)
    
#     return reference_mfcc

# def similarity(input_features, reference_features):
#     """計算輸入語音與參考語音之間的相似度"""
#     # 如果輸入特徵是三維的，將其展平為二維
#     if input_features.ndim == 3:
#         input_features = input_features.reshape(input_features.shape[0], -1)
#     return cosine_similarity(input_features, reference_features)


# def main():
#     # 標準語音檔案路徑
#     wav_filename = r"D:\wsad2314666\Alcoho.github.io\語音評分系統0510\A.wav"

#     # 讀取標準語音檔案並計算MFCC特徵
#     reference_mfcc = load_reference_data(wav_filename)

#     # 錄製與標準語音相同長度的使用者輸入語音
#     user_input_filename = "user_input.wav"
#     record_audio(user_input_filename)

#     # 讀取使用者輸入的語音檔案並計算MFCC特徵
#     rate_user, data_user = load_wav(user_input_filename)
#     input_mfcc = calculate_mfcc(data_user, rate_user)

#     # 計算使用者輸入語音與標準語音的相似度
#     sim_scores = similarity(input_mfcc, reference_mfcc)

#     print("使用者輸入語音與標準語音的相似度比對結果：", sim_scores)
#     score=100/1+2000*(sim_scores)**2
#     print("語音評分結果：", score)
#     # # 將相似度比對結果繪製成圖像
#     # plt.imshow(score, cmap='viridis', interpolation='nearest')
#     # plt.colorbar(label='Similarity')
#     # plt.xlabel('Standard Speech')
#     # plt.ylabel('User Input')
#     # plt.title('Similarity Matrix')
#     # plt.show()

#     plt.plot(score)
#     plt.xlabel('Frame')
#     plt.ylabel('Score')
#     plt.title('Speech Similarity Score')
#     plt.show()

# if __name__ == "__main__":
#     main()

# 正常程式
# import numpy as np
# import librosa
# import sounddevice as sd
# from scipy.ndimage import maximum_filter1d
# import matplotlib.pyplot as plt
# # 載入音檔
# def load_audio(file_path):
#     audio, sr = librosa.load(file_path, sr=None)
#     return audio, sr

# # 提取 MFCC 特徵
# def extract_mfcc(audio, sr):
#     mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
#     return mfccs

# # 正規化 MFCC 特徵
# def normalize_mfcc(mfccs):
#     return (mfccs - np.mean(mfccs)) / np.std(mfccs)

# # 計算相似度分數
# def compute_similarity_score(mfccs_A, mfccs_B):
#     return np.mean(np.abs(mfccs_A - mfccs_B)) * 100

# # 去除靜音部分
# def remove_silence(audio, threshold=0.02):
#     non_silent_intervals = librosa.effects.split(audio, top_db=threshold)
#     non_silent_audio = np.concatenate([audio[start:end] for start, end in non_silent_intervals])
#     return non_silent_audio

# # 預處理雜音
# def preprocess_audio(audio):
#     # 雜音預處理，例如濾波等
#     # 這裡我們使用 maximum_filter1d 函數來濾波
#     filtered_audio = maximum_filter1d(audio, size=3)
#     return filtered_audio

# # 主函式
# def main(audio_file_A,audio_file_B):
#     # 載入音檔 A
#     audio_A, sr_A = load_audio(audio_file_A)
#     # 載入音檔 B
#     audio_B, sr_B = load_audio(audio_file_B)
#     # # 錄製音檔 B
#     # print("請開始錄音...")
#     # audio_B = sd.rec(len(audio_A), samplerate=sr_A, channels=1, dtype='float32')
#     # sd.wait()

#     # 去除靜音部分
#     audio_A = remove_silence(audio_A)
#     audio_B = remove_silence(audio_B)
#     # audio_B = remove_silence(audio_B[:, 0])  # 取錄製的音檔的第一個聲道

#     # 預處理雜音
#     audio_A = preprocess_audio(audio_A)
#     audio_B = preprocess_audio(audio_B)

#     # 提取 MFCC 特徵
#     mfccs_A = extract_mfcc(audio_A, sr_A)
#     mfccs_B = extract_mfcc(audio_B, sr_A)

#     # 使兩個音檔的 MFCC 特徵具有相同的維度
#     min_length = min(mfccs_A.shape[1], mfccs_B.shape[1])
#     mfccs_A = mfccs_A[:, :min_length]
#     mfccs_B = mfccs_B[:, :min_length]

#     # 正規化 MFCC 特徵
#     mfccs_A_normalized = normalize_mfcc(mfccs_A)
#     mfccs_B_normalized = normalize_mfcc(mfccs_B)

#    # 繪製 MFCC 圖形
#     plt.figure(figsize=(10, 4))
#     plt.subplot(2, 1, 1)
#     librosa.display.specshow(mfccs_A_normalized, x_axis='time')
#     plt.colorbar()
#     plt.title('MFCC of Audio A')
#     plt.ylabel('MFCC Coefficients')
#     plt.xlabel('Time')

#     plt.subplot(2, 1, 2)
#     librosa.display.specshow(mfccs_B_normalized, x_axis='time')
#     plt.colorbar()
#     plt.title('MFCC of Audio B')
#     plt.ylabel('MFCC Coefficients')
#     plt.xlabel('Time')

#     plt.tight_layout()
#     plt.show()

#     # 計算相似度分數
#     score = 100-compute_similarity_score(mfccs_A_normalized, mfccs_B_normalized)

#     return score
# # 範例使用
# if __name__ == "__main__":
#     audio_file_A = r"C:\Users\Cmsh\Desktop\語音評分\A.wav"
#     audio_file_B = r"C:\Users\Cmsh\Desktop\語音評分\output.wav"
#     similarity_score = main(audio_file_A,audio_file_B)
#     print("相似度分數:", similarity_score)
# sk-proj-GuDkyZWoOKeBwzSw0HTVT3BlbkFJM1S6hhwiv7PYrKmljWJf   gpt API