import os
import librosa
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model,Model
from tensorflow.keras.layers import Input,Lambda,LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import logging
import pickle
logging.basicConfig(level=logging.INFO)

def audio_to_mfcc(audio_path, sr=44100,max_len=87):
    y, sr = librosa.load(audio_path, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    mean = np.mean(mfcc, axis=1, keepdims=True)
    cms_mfccs = mfcc - mean
    if mfcc.shape[1] > max_len:
        mfcc = mfcc[:, :max_len]
    else:
        pad_width = max_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    mfcc_length = mfcc.shape[1]  # 实际 MFCC 长度
    return cms_mfccs.T, mfcc_length  # Transpose to shape (time, features)
def save_mfcc_array(mfcc, output_path):
    with open(output_path, 'wb') as f:
        pickle.dump(mfcc, f)
def process_audio_files(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    lengths = []
    for audio_file in os.listdir(input_dir):
        if audio_file.endswith('.wav'):
            audio_path = os.path.join(input_dir, audio_file)
            mfcc, length = audio_to_mfcc(audio_path)
            lengths.append(length)  # 保存長度
            output_path = os.path.join(output_dir, f"{os.path.splitext(audio_file)[0]}.pkl")
            with open(output_path, 'wb') as f:
                pickle.dump(mfcc, f)
            logging.info(f"Processed {audio_file} to {output_path}")
    return lengths
def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    
    # 确保所有输入都是浮点类型
    y_pred = tf.cast(y_pred, tf.float32)
    labels = tf.cast(labels, tf.float32)
    input_length = tf.cast(input_length, tf.float32)
    label_length = tf.cast(label_length, tf.float32)

    def body(y_pred, labels, input_length, label_length, loss):
        loss = tf.keras.backend.ctc_batch_cost(labels, y_pred, input_length, label_length)
        return [y_pred, labels, input_length, label_length, loss]

    initial_loss = tf.zeros_like(input_length, dtype=tf.float32)
    loop_vars = [y_pred, labels, input_length, label_length, initial_loss]

    shape_invariants = [
        tf.TensorShape([None, None, None]),
        tf.TensorShape([None, None]),
        tf.TensorShape([None, 1]),
        tf.TensorShape([None, 1]),
        tf.TensorShape([None, 1])
    ]

    _, _, _, _, loss = tf.while_loop(
        cond=lambda *_: True,
        body=body,
        loop_vars=loop_vars,
        shape_invariants=shape_invariants
    )

    return loss
def create_model(num_classes):
    input_data = Input(name='the_input', shape=(None, 20), dtype='float32')
    labels = Input(name='the_labels', shape=(None,), dtype='float32')
    input_length = Input(name='input_length', shape=(1,), dtype='float32')
    label_length = Input(name='label_length', shape=(1,), dtype='float32')

    x = LSTM(128, return_sequences=True)(input_data)
    x = LSTM(64, return_sequences=True)(x)
    y_pred = Dense(num_classes + 1, activation='softmax', name='y_pred')(x)

    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])

    model = Model(inputs=[input_data, labels, input_length, label_length], outputs=loss_out)
    model.compile(optimizer=Adam(learning_rate=0.0001), loss={'ctc': lambda y_true, y_pred: y_pred})
    return model


def load_data(data_dir, max_len=87):
    X, y = [], []
    input_lengths, label_lengths = [], []
    for file in os.listdir(data_dir):
        if file.endswith('.pkl'):
            with open(os.path.join(data_dir, file), 'rb') as f:
                mfcc = pickle.load(f)
                if mfcc.shape[0] == max_len:  # 确保 MFCC 特征长度为 87
                    X.append(mfcc)
                    y.append(1 if 'positive' in file else 0)
                    input_lengths.append(mfcc.shape[0])  # 假设输入长度等于 MFCC 特征长度
                    label_lengths.append(1)  # 假设每个样本的标签长度为 1，根据实际情况调整
                else:
                    print(f"Skipping {file} due to incorrect shape: {mfcc.shape}")
    
    # 将列表转换为 NumPy 数组
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    input_lengths = np.array(input_lengths, dtype=np.float32)
    label_lengths = np.array(label_lengths, dtype=np.float32)
    
    # 确保 y 是二维的
    y = np.expand_dims(y, axis=-1)
    
    print(f"Final data shapes: X={X.shape}, y={y.shape}")
    return {
        'X': X, 
        'y': y, 
        'input_lengths': input_lengths, 
        'label_lengths': label_lengths
    }

def train_model(train_dir, model_path='LSTM_model.keras', epochs=2, batch_size=32):
    try:
        data = load_data(train_dir)
        X = data['X']
        y = data['y']
        input_lengths = np.expand_dims(data['input_lengths'], axis=-1)
        label_lengths = np.expand_dims(data['label_lengths'], axis=-1)

        num_classes = 32  # 根据实际情况调整类别数
        model = create_model(num_classes)

        logging.info("Starting model training...")
        history = model.fit(
            [X, y, input_lengths, label_lengths], y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            callbacks=[
                tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001),
                tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            ],
            verbose=1  # 添加这行以显示训练进度
        )

        model.save(model_path)
        logging.info(f"Model saved to {model_path}")
        return history
    except Exception as e:
        logging.error(f"An error occurred during training: {str(e)}")
        raise 

def process_test_audio(audio_path, model_path='LSTM_model.keras'):
    mfcc = audio_to_mfcc(audio_path)
    mfcc = np.expand_dims(mfcc, axis=0)  # Add batch dimension
    
    model = load_model(model_path)
    
    prediction = model.predict(mfcc)
    
    return prediction[0][0]

def main():
    input_dir = r"D:\wsad231466\Alcoho.github.io\flask-templete\train"
    output_dir = r"D:\wsad231466\Alcoho.github.io\flask-templete\mfcc_arrays"
    
    process_audio_files(input_dir, output_dir)

    # 训练阶段
    try:
        logging.info("Starting model training...")
        history = train_model(output_dir, epochs=2, batch_size=32)
        logging.info("Training completed successfully")
        
        # 打印训练历史
        logging.info(f"Training history: {history.history}")
    except Exception as e:
        logging.error(f"An error occurred during training: {str(e)}")
        return

    test_audio_path = r"D:\wsad231466\Alcoho.github.io\flask-templete\static\audio\user_input.wav"
    model_path = 'LSTM_model.keras'

    try:
        result = process_test_audio(test_audio_path, model_path)
        logging.info(f"Test audio raw prediction: {result}")
        logging.info(f"Test audio prediction result: {result * 100:.2f}%")
    except Exception as e:
        logging.error(f"An error occurred during prediction for test audio: {str(e)}")

if __name__ == "__main__":
    main()