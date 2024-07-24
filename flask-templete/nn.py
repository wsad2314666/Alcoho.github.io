import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import backend as K
import sounddevice as sd
import wavio

# 設置參數
n_mfcc = 20
max_length = 44100
sample_rate = 44100
num_classes = 27  # Including the blank label for CTC

# 加載音檔並轉換為MFCC
def load_and_convert_to_mfcc(file_path):
    y, sr = librosa.load(file_path, sr=sample_rate, duration=1.0)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    if mfcc.shape[1] < max_length:
        mfcc = np.pad(mfcc, ((0, 0), (0, max_length - mfcc.shape[1])), mode='constant')
    else:
        mfcc = mfcc[:, :max_length]
    return mfcc.T

# 儲存MFCC圖片
def save_mfcc_image(mfcc, file_path):
    plt.figure(figsize=(6, 4))
    librosa.display.specshow(mfcc.T, sr=sample_rate, x_axis='time')
    plt.colorbar()
    plt.title('MFCC')
    plt.tight_layout()
    plt.savefig(file_path)
    plt.close()

# 準備訓練數據
def prepare_data(file_paths):
    data = []
    for file_path in file_paths:
        mfcc = load_and_convert_to_mfcc(file_path)
        data.append(mfcc)
    data = np.array(data)
    data = data[..., np.newaxis]
    return data

# 創建模型
def create_model(input_shape):
    inp = tf.keras.Input(shape=input_shape)
    inp_len = tf.keras.Input(shape=(1,), dtype='int32')
    seq_len = tf.keras.Input(shape=(1,), dtype='int32')
    
    x = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(inp)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    
    x = tf.keras.layers.Reshape((-1, x.shape[-1] * x.shape[-2]))(x)  # Flatten the output to (time, features)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(128, return_sequences=True))(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(num_classes, activation='softmax'))(x)
    
    model = tf.keras.models.Model(inputs=[inp, inp_len, seq_len], outputs=x)
    return model

# 定義CTC損失函數
def ctc_loss(y_true, y_pred):
    label_length = tf.math.count_nonzero(y_true, axis=-1, keepdims=True, dtype='int32')
    input_length = tf.fill([tf.shape(y_pred)[0], 1], tf.shape(y_pred)[1])
    loss = K.ctc_batch_cost(y_true, y_pred, input_length, label_length)
    return loss

# 訓練模型
def train_model(model, train_data, train_labels, train_inp_lengths, train_seq_lengths):
    model.compile(loss=ctc_loss, optimizer='Adam')
    model.fit([train_data, train_inp_lengths, train_seq_lengths], train_labels, batch_size=8, epochs=20)

# 錄製音頻
def record_audio(file_path, duration=1.0, fs=44100):
    print("Recording...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    wavio.write(file_path, recording, fs, sampwidth=2)
    print("Recording finished.")

# 主程式
if __name__ == "__main__":
    # 輸入音檔路徑
    file_paths = ["C:\\Users\\USER\\Desktop\\flask-templete\\train\\A1.wav", "C:\\Users\\USER\\Desktop\\flask-templete\\train\\A2.wav", "C:\\Users\\USER\\Desktop\\flask-templete\\train\\A3.wav", "C:\\Users\\USER\\Desktop\\flask-templete\\train\\A4.wav", "C:\\Users\\USER\\Desktop\\flask-templete\\train\\A5.wav",
                  "C:\\Users\\USER\\Desktop\\flask-templete\\train\\A6.wav", "C:\\Users\\USER\\Desktop\\flask-templete\\train\\A7.wav", "C:\\Users\\USER\\Desktop\\flask-templete\\train\\A8.wav", "C:\\Users\\USER\\Desktop\\flask-templete\\train\\A9.wav", "C:\\Users\\USER\\Desktop\\flask-templete\\train\\A10.wav"]
    
    # 準備訓練數據
    train_data = prepare_data(file_paths)

    # 生成隨機標籤，確保標籤數據中不包含非法值
    train_labels = np.random.randint(1, num_classes - 1, (10, 10))  # 使用1到num_classes-2之間的值
    train_labels[:, -1] = 0  # 確保最後一個標籤是空白標籤

    train_inp_lengths = np.full((10, 1), max_length)
    train_seq_lengths = np.full((10, 1), 10)

    # 創建模型
    input_shape = (max_length, n_mfcc, 1)
    model = create_model(input_shape)

    # 訓練模型
    train_model(model, train_data, train_labels, train_inp_lengths, train_seq_lengths)
    
    # 錄製並轉換音頻
    record_file = "C:\\Users\\USER\\Desktop\\flask-templete\\static\\audio\\user_input.wav"
    record_audio(record_file)
    recorded_mfcc = load_and_convert_to_mfcc(record_file)
    save_mfcc_image(recorded_mfcc, "user_input.png")
    
    # 預測並比對結果
    recorded_mfcc = recorded_mfcc[np.newaxis, ..., np.newaxis]
    predictions = model.predict([recorded_mfcc, np.array([[max_length]]), np.array([[10]])])

    # 解码预测结果
    decoded_pred = K.ctc_decode(predictions, input_length=np.array([[max_length]]))[0][0]
    dense_pred = tf.sparse.to_dense(decoded_pred).numpy()

    # 假设你有一个实际标签 true_labels
    true_labels = np.array([1, 2, 3, 4, 5, 0])  # 示例标签

    # 计算匹配百分比
    pred_text = dense_pred[0]
    match_count = sum(1 for p, t in zip(pred_text, true_labels) if p == t)
    accuracy = (match_count / len(true_labels)) * 100

    print(f"Prediction Accuracy: {accuracy:.2f}%")