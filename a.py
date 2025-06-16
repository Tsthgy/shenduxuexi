import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler, MaxAbsScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.calibration import CalibratedClassifierCV
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import make_pipeline
import joblib
import time
import os
os.environ["PYTHONUTF8"] = "1"

# 配置参数
class Config:
    SEED = 42
    N_JOBS = 1

    TEST_SIZE = 0.2
    N_SPLITS = 5
    MAX_FEATURES = 100000
    NGRAM_RANGE = (1, 3)
    TOP_K_FEATURES = 50000


# 标签映射
label_map = {'科技': 0, '股票': 1, '体育': 2, '娱乐': 3, '时政': 4,'社会': 5, '教育': 6, '财经': 7, '家居': 8, '游戏': 9,'房产': 10, '时尚': 11, '彩票': 12, '星座': 13}

def cross_validate_model(model, X, y):
    # 设置 n_jobs=1 并添加错误处理
    try:
        scores = cross_val_score(
            model, X, y, cv=5, scoring='accuracy', n_jobs=1
        )
    except UnicodeEncodeError:
        print("编码错误，尝试使用单进程和 UTF-8 编码")
        scores = cross_val_score(
            model, X, y, cv=5, scoring='accuracy', n_jobs=1
        )
    return scores.mean()
# 自定义特征提取器
class TextStats(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return pd.DataFrame({
            'text_length': X.apply(len),
            'num_unique': X.apply(lambda x: len(set(x.split())))
        })


# 1. 增强的数据加载
def load_data(train_path, test_path):
    train_df = pd.read_csv(train_path, sep='\t', dtype={'text': str})
    test_df = pd.read_csv(test_path, sep='\t', dtype={'text': str})

    # 数据质量检查
    print(f"训练集样本数: {len(train_df)}")
    print(f"测试集样本数: {len(test_df)}")
    print("类别分布:\n", train_df['label'].value_counts(normalize=True))

    return train_df, test_df


# 2. 改进的数据预处理
def preprocess_data(train_df, test_df):
    # 文本清洗
    train_df['text'] = train_df['text'].str.replace(r'\d+', ' ')  # 简单清洗
    test_df['text'] = test_df['text'].str.replace(r'\d+', '' )

    # 分层分割训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(
        train_df['text'],
        train_df['label'],
        test_size=Config.TEST_SIZE,
        random_state=Config.SEED,
        stratify=train_df['label']
    )

    return X_train, X_val, y_train, y_val, test_df['text']


# 3. 高级特征工程
def build_feature_union():
    # TF-IDF特征
    tfidf = TfidfVectorizer(
        token_pattern=r'\w+',
        ngram_range=Config.NGRAM_RANGE,
        max_features=Config.TOP_K_FEATURES,
        sublinear_tf=True
    )

    # 哈希特征
    hashing = HashingVectorizer(
        ngram_range=Config.NGRAM_RANGE,
        n_features=Config.MAX_FEATURES,
        alternate_sign=False
    )

    # 特征组合
    feature_union = FeatureUnion([
        ('tfidf', tfidf),
        ('hashing', hashing),
        ('text_stats', TextStats())
    ])

    return feature_union


# 4. 增强的模型构建
def build_model():
    feature_union = build_feature_union()

    # 使用校准的LinearSVC
    base_model = LinearSVC(
        C=0.5,
        class_weight='balanced',
        dual=False,
        max_iter=2000,
        random_state=Config.SEED
    )

    # 校准模型
    calibrated_model = CalibratedClassifierCV(base_model, cv=3)

    # 构建完整管道
    model = make_pipeline(
        feature_union,
        MaxAbsScaler(),
        RandomOverSampler(random_state=Config.SEED),
        calibrated_model
    )

    return model


# 5. 交叉验证评估
def cross_validate_model(model, X, y):
    cv = StratifiedKFold(n_splits=Config.N_SPLITS, shuffle=True, random_state=Config.SEED)
    scores = cross_val_score(
        model, X, y,
        cv=cv,
        scoring='f1_macro',
        n_jobs=Config.N_JOBS
    )
    print(f"交叉验证F1分数: {np.mean(scores):.4f} (±{np.std(scores):.4f})")
    return np.mean(scores)


# 6. 模型训练和评估
def train_and_evaluate(model, X_train, y_train, X_val, y_val):
    print("开始训练模型...")
    start_time = time.time()

    model.fit(X_train, y_train)

    print(f"训练完成，耗时: {time.time() - start_time:.2f}秒")

    # 验证集评估
    y_pred = model.predict(X_val)
    score = f1_score(y_val, y_pred, average='macro')
    print(f"验证集F1分数: {score:.4f}")
    print("\n分类报告:\n", classification_report(y_val, y_pred))

    return model, score


# 7. 生成预测结果
def generate_predictions(model, test_text):
    print("生成测试集预测...")
    return model.predict(test_text)


# 8. 保存结果
def save_results(test_df, predictions, output_path):
    result = pd.DataFrame({'label': predictions})
    result.to_csv(output_path, index=False)
    print(f"结果已保存到 {output_path}")


# 主函数
def main():
    # 文件路径
    train_path = './input/train_set.csv'
    test_path = './input/test_a.csv'
    output_path = './submission.csv'

    # 加载数据
    train_df, test_df = load_data(train_path, test_path)

    # 预处理数据
    X_train, X_val, y_train, y_val, test_text = preprocess_data(train_df, test_df)

    # 构建模型
    model = build_model()

    # 交叉验证
    print("进行交叉验证...")
    cv_score = cross_validate_model(model, X_train, y_train)

    # 训练和评估
    model, val_score = train_and_evaluate(model, X_train, y_train, X_val, y_val)

    # 生成预测
    predictions = generate_predictions(model, test_text)

    # 保存结果
    save_results(test_df, predictions, output_path)

    # 保存模型
    joblib.dump(model, 'optimized_text_classifier.joblib')
    print("模型已保存")


if __name__ == '__main__':
    main()