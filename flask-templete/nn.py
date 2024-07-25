import os
import librosa
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.sequence import pad_sequences

def extract_and_normalize_features(file_path):
    y, sr = librosa.load(file_path, sr=44100)
    melspectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    cmvn = (melspectrogram - np.mean(melspectrogram, axis=1, keepdims=True)) / (np.std(melspectrogram, axis=1, keepdims=True) + 1e-6)
    return cmvn.T

def create_transformer_model(input_shape, vocab_size):
    input_data = layers.Input(name='input', shape=input_shape, dtype='float32')
    x = layers.Masking(mask_value=0.0)(input_data)
    x = layers.Conv1D(64, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(128, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    transformer_block = layers.MultiHeadAttention(num_heads=8, key_dim=128)(x, x)
    transformer_block = layers.LayerNormalization(epsilon=1e-6)(transformer_block)
    transformer_block = layers.Dense(256, activation='relu')(transformer_block)
    transformer_block = layers.Dropout(0.2)(transformer_block)
    transformer_block = layers.Dense(vocab_size + 1, activation='softmax')(transformer_block)
    
    model = tf.keras.Model(inputs=input_data, outputs=transformer_block)
    return model

def data_generator(audio_files, transcripts, batch_size):
    max_length = 0
    for file in audio_files:
        features = extract_and_normalize_features(file)
        max_length = max(max_length, features.shape[0])
    
    print(f"最大长度: {max_length}")

    while True:
        x_batch = []
        y_batch = []
        for start in range(0, len(audio_files), batch_size):
            batch_files = audio_files[start:start+batch_size]
            batch_transcripts = transcripts[start:start+batch_size]
            for file, transcript in zip(batch_files, batch_transcripts):
                features = extract_and_normalize_features(file)
                padded_features = np.pad(features, ((0, max_length - features.shape[0]), (0, 0)), mode='constant')
                x_batch.append(padded_features)
                y_batch.append([1] * features.shape[0] + [0] * (max_length - features.shape[0]))
            
            print(f"批次形状: x={np.array(x_batch).shape}, y={np.array(y_batch).shape}")
            yield np.array(x_batch), np.array(y_batch)
            x_batch.clear()
            y_batch.clear()

def calculate_wer(true_transcript, pred_transcript):
    true_words = true_transcript.split()
    pred_words = pred_transcript.split()
    S = sum(1 for true_word, pred_word in zip(true_words, pred_words) if true_word != pred_word)
    D = len(true_words) - len(pred_words)
    I = len(pred_words) - len(true_words)
    return (S + max(D, 0) + max(I, 0)) / len(true_words)

def evaluate_model(model, history, test_files, test_transcripts, max_length, predictions):
    # 评估模型在训练数据上的性能
    train_loss = history.history['loss'][-1]
    print(f"Training loss: {train_loss}")

    # 评估模型在测试数据上的性能
    total_wer = 0
    for true_transcript, pred_transcript in zip(test_transcripts, predictions):
        print(f"True transcript: {true_transcript}")
        print(f"Predicted transcript: {pred_transcript}")
        
        # 计算 WER
        wer = calculate_wer(true_transcript, pred_transcript)
        print(f"Word Error Rate: {wer}")
        
        total_wer += wer
    
    avg_wer = total_wer / len(test_files)
    print(f"Average Word Error Rate: {avg_wer}")
    
    return train_loss, avg_wer


def ctc_loss_function(y_true, y_pred):
    y_true = tf.cast(y_true, 'int32')
    input_length = tf.fill([tf.shape(y_pred)[0]], tf.shape(y_pred)[1])
    label_length = tf.reduce_sum(tf.cast(y_true != 0, 'int32'), axis=-1)
    loss = tf.nn.ctc_loss(
        labels=y_true,
        logits=y_pred,
        label_length=label_length,
        logit_length=input_length,
        logits_time_major=False,
        blank_index=0
    )
    return tf.reduce_mean(loss)

def main():
    base_path = "D:\\wsad231466\\Alcoho.github.io\\flask-templete\\train\\"
    audio_files = [os.path.join(base_path, f"A{i}.wav") for i in range(1, 54)]  # 生成A1到A53的音檔路徑
    test_files = ["D:\\wsad231466\\Alcoho.github.io\\flask-templete\\static\\audio\\user_input.wav"]
    transcripts = ["A"] * len(audio_files)
    
    print(f"音频文件数量: {len(audio_files)}")
    
    # 获取最大长度
    max_length = 0
    for file in audio_files + test_files:
        features = extract_and_normalize_features(file)
        max_length = max(max_length, features.shape[0])
    
    print(f"最大长度: {max_length}")
    
    # 设置输入形状和词汇表大小
    input_shape = (max_length, 128)  # 根据Mel频谱图特征数量设置
    vocab_size = 1  # 因为我们只有一个标签"A"
    
    model = create_transformer_model(input_shape, vocab_size)
    model.compile(optimizer='adam', loss=ctc_loss_function)

    batch_size = 16
    steps_per_epoch = len(audio_files) // batch_size

    print(f"每轮步数: {steps_per_epoch}")

    # 保存预测结果
    predictions = []

    for epoch in range(20):
        history = model.fit(data_generator(audio_files, transcripts, batch_size), 
                            steps_per_epoch=steps_per_epoch, 
                            epochs=1,
                            verbose=1)
        
        # 保存每个epoch的预测结果
        for file in test_files:
            features = extract_and_normalize_features(file)
            padded_features = np.pad(features, ((0, max_length - features.shape[0]), (0, 0)), mode='constant')
            pred_prob = model.predict(np.expand_dims(padded_features, axis=0), verbose=0)
            pred_sequence = np.argmax(pred_prob, axis=-1)
            pred_transcript = "".join(["A" if p == 1 else "" for p in pred_sequence[0]])
            predictions.append(pred_transcript)

    print("训练历史:")
    print(history.history)

    train_loss, avg_wer = evaluate_model(model, history, test_files, ["A"], max_length, predictions)
    wcr = 1 - avg_wer
    print(f'Training loss: {train_loss}')
    print(f'Average Word Error Rate: {avg_wer}')
    print(f'Word Correct Rate: {wcr * 100:.2f}%')

if __name__ == "__main__":
    main()