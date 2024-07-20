import os
import numpy as np
import librosa
import tensorflow as tf
import wave
import pyaudio
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

# 資料路徑
train_audio_path = 'C:\\Users\\USER\\Desktop\\flask-templete\\train'
test_audio_path = 'C:\\Users\\USER\\Desktop\\flask-templete\\static\\audio'

def extract_features(y, sr):
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    mel = librosa.feature.melspectrogram(y=y, sr=sr)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    
    mfccs = np.mean(mfccs.T, axis=0)
    chroma = np.mean(chroma.T, axis=0)
    mel = np.mean(mel.T, axis=0)
    contrast = np.mean(contrast.T, axis=0)
    
    return np.hstack([mfccs, chroma, mel, contrast])

def load_audio_files(path, num_files, duration=3):
    audio_files = []
    labels = []
    for i in range(1, num_files + 1):
        file_path = os.path.join(path, f'A{i}.wav')
        y, sr = librosa.load(file_path, sr=None, duration=duration)
        if len(y) < sr * duration:
            y = np.pad(y, (0, sr * duration - len(y)))
        features = extract_features(y, sr)
        audio_files.append(features)
        labels.append(0)  # 假設所有訓練檔案標籤相同
    return np.array(audio_files), np.array(labels)

def compute_similarity(model, test_file, duration=3):
    y, sr = librosa.load(test_file, sr=None, duration=duration)
    if len(y) < sr * duration:
        y = np.pad(y, (0, sr * duration - len(y)))
    features = extract_features(y, sr)
    features = np.expand_dims(features, axis=0)  # 擴展維度以符合模型輸入
    prediction = model.predict(features)
    return prediction[0][0]

# 構建神經網路模型
def create_model(input_shape):
    model = Sequential()
    model.add(Dense(256, activation='relu', input_shape=input_shape))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))  # 二元分類
    model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])
    return model

# 訓練模型
def train_model(model, x_train, y_train, epochs=50, batch_size=8):
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
    return model
def record_audio_to_file(filename, duration=3, channels=1, rate=44100, frames_per_buffer=1):
    """Record user's input audio and save it to the specified file."""
    FORMAT = pyaudio.paInt16
    p = pyaudio.PyAudio()

    # Open stream
    stream = p.open(format=FORMAT, channels=channels, rate=rate, frames_per_buffer=frames_per_buffer, input=True)
    print("Start recording...")

    frames = []
    # Record for the given duration
    for _ in range(0, int(rate / frames_per_buffer * duration)):
        data = stream.read(frames_per_buffer)
        frames.append(data)
    print("Recording stopped")
    stream.stop_stream()
    stream.close()
    p.terminate()
    # Save the recorded data as a WAV file
    wf = wave.open(filename, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(rate)
    wf.writeframes(b''.join(frames))
    wf.close()

    print(f"Audio recorded and saved to {filename}")
    return filename
# 主程式
def main():
    # 載入訓練資料
    x_train, y_train = load_audio_files(train_audio_path, 10)
    input_shape = (x_train.shape[1],)
    
    # 創建並訓練模型
    model = create_model(input_shape)
    model = train_model(model, x_train, y_train)

    #錄製音檔    
    #record_audio_to_file('C:\\Users\\USER\\Desktop\\flask-templete\\static\\audio\\user_input.wav')
    # 測試音檔相似度計算
    test_file = os.path.join(test_audio_path, 'A.wav')
    similarity = compute_similarity(model, test_file)
    print(f'Similarity with training audio: {similarity:.2f}')

if __name__ == "__main__":
    main()
