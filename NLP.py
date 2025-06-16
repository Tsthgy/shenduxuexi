import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import joblib

# 标签映射
label_map = {'科技': 0, '股票': 1, '体育': 2, '娱乐': 3, '时政': 4,
             '社会': 5, '教育': 6, '财经': 7, '家居': 8, '游戏': 9,
             '房产': 10, '时尚': 11, '彩票': 12, '星座': 13}


# 1. 数据加载
def load_data(train_path, test_path):
    train_df = pd.read_csv(train_path, sep='\t')
    test_df = pd.read_csv(test_path, sep='\t')
    return train_df, test_df


# 2. 数据预处理
def preprocess_data(train_df, test_df):
    # 将文本转换为字符串格式
    train_df['text'] = train_df['text'].astype(str)
    test_df['text'] = test_df['text'].astype(str)

    # 分割训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(
        train_df['text'], train_df['label'], test_size=0.2, random_state=42)

    return X_train, X_val, y_train, y_val, test_df['text']


# 3. 特征工程和模型构建
def build_model():
    # 使用TF-IDF进行文本特征提取
    tfidf = TfidfVectorizer(
        token_pattern=r'\d+',  # 匹配数字作为token
        ngram_range=(1, 2),  # 使用1-2 gram
        max_features=50000,  # 限制特征数量
        sublinear_tf=True  # 使用sublinear tf scaling
    )

    # 使用线性SVC分类器
    classifier = LinearSVC(
        C=1.0,
        class_weight='balanced',
        dual=False,
        max_iter=1000,
        random_state=42
    )

    # 构建管道
    model = Pipeline([
        ('tfidf', tfidf),
        ('classifier', classifier)
    ])

    return model


# 4. 模型训练和评估
def train_and_evaluate(model, X_train, y_train, X_val, y_val):
    print("开始训练模型...")
    model.fit(X_train, y_train)

    print("在验证集上评估模型...")
    y_pred = model.predict(X_val)
    score = f1_score(y_val, y_pred, average='macro')
    print(f"验证集F1分数: {score:.4f}")

    return model


# 5. 生成预测结果
def generate_predictions(model, test_text):
    print("生成测试集预测...")
    predictions = model.predict(test_text)
    return predictions


# 6. 保存结果
def save_results(test_df, predictions, output_path):
    result = pd.DataFrame()
    result['label'] = predictions
    result.to_csv(output_path, index=False)
    print(f"结果已保存到 {output_path}")


# 主函数
def main():
    # 文件路径
    train_path = './input/train_set.csv'
    test_path = './input/test_a.csv'
    sample_path = './input/test_a_sample_submit.csv'
    output_path = './submission1.csv'

    # 加载数据
    train_df, test_df = load_data(train_path, test_path)

    # 预处理数据
    X_train, X_val, y_train, y_val, test_text = preprocess_data(train_df, test_df)

    # 构建模型
    model = build_model()

    # 训练和评估模型
    model = train_and_evaluate(model, X_train, y_train, X_val, y_val)

    # 生成预测
    predictions = generate_predictions(model, test_text)

    # 保存结果
    save_results(test_df, predictions, output_path)

    # 可选：保存模型
   # joblib.dump(model, 'text_classifier.joblib')


if __name__ == '__main__':
    main()