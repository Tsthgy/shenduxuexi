import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical


# 1. 数据加载和预处理
def load_data():
    # 加载训练数据
    train_df = pd.read_csv('./input/train_set.csv', sep='\t')

    # 加载测试数据（根据实际文件名调整）
    test_df = pd.read_csv('./input/test_a.csv', sep='\t')

    return train_df, test_df


# 2. 数据预处理
def preprocess_data(train_df, test_df, max_words=50000, max_len=500):
    # 合并训练和测试文本数据以便统一tokenizer
    texts = train_df['text'].tolist() + test_df['text'].tolist()

    # 初始化tokenizer
    tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')
    tokenizer.fit_on_texts(texts)

    # 将文本转换为序列
    train_sequences = tokenizer.texts_to_sequences(train_df['text'])
    test_sequences = tokenizer.texts_to_sequences(test_df['text'])

    # 填充序列到相同长度
    X_train = pad_sequences(train_sequences, maxlen=max_len, padding='post', truncating='post')
    X_test = pad_sequences(test_sequences, maxlen=max_len, padding='post', truncating='post')

    # 准备标签数据
    y_train = to_categorical(train_df['label'])

    return X_train, y_train, X_test, tokenizer


# 3. 构建深度学习模型
def build_model(max_words, max_len, num_classes):
    model = Sequential([
        Embedding(input_dim=max_words, output_dim=128, input_length=max_len),
        Bidirectional(LSTM(64, return_sequences=True)),
        Dropout(0.5),
        Bidirectional(LSTM(32)),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


# 4. 训练模型
def train_model(X_train, y_train, max_words, max_len, num_classes):
    # 划分训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42)

    # 构建模型
    model = build_model(max_words, max_len, num_classes)

    # 设置回调函数
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=3),
        ModelCheckpoint('best_model.h5', save_best_only=True)
    ]

    # 训练模型
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=10,
        batch_size=128,
        callbacks=callbacks
    )

    return model


# 5. 评估模型
def evaluate_model(model, X_val, y_val):
    # 预测验证集
    y_pred = model.predict(X_val)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_val, axis=1)

    # 计算F1分数
    f1 = f1_score(y_true, y_pred_classes, average='macro')
    print(f'Validation F1 Score: {f1:.4f}')

    return f1


# 6. 生成预测结果
def generate_submission(model, X_test, test_df):
    # 预测测试集
    y_test_pred = model.predict(X_test)
    y_test_pred_classes = np.argmax(y_test_pred, axis=1)

    # 创建提交文件
    submission = pd.DataFrame({
        'label': y_test_pred_classes
    }, index=test_df.index)

    # 保存为CSV文件
    submission.to_csv('submission.csv', index=False)
    print("Submission file saved as 'submission.csv'")


# 主函数
def main():
    # 参数设置
    MAX_WORDS = 50000  # 词汇表大小
    MAX_LEN = 500  # 文本最大长度
    NUM_CLASSES = 14  # 类别数量

    # 1. 加载数据
    train_df, test_df = load_data()

    # 2. 数据预处理
    X_train, y_train, X_test, tokenizer = preprocess_data(
        train_df, test_df, MAX_WORDS, MAX_LEN)

    # 3. 训练模型
    model = train_model(X_train, y_train, MAX_WORDS, MAX_LEN, NUM_CLASSES)

    # 4. 评估模型（使用验证集）
    # 这里需要从训练数据中划分验证集
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42)
    evaluate_model(model, X_val, y_val)

    # 5. 生成预测结果
    generate_submission(model, X_test, test_df)


if __name__ == '__main__':
    main()