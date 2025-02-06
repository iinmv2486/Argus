import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier

def calculate_feature_importances(input_csv, output_file, batch_size=10000):
    """
    주어진 데이터셋에서 Feature Importance를 계산하고 결과를 파일로 저장합니다.

    Parameters
    ----------
    input_csv : str
        입력 데이터 파일 경로 (CSV 형식)
    output_file : str
        Feature 중요도를 저장할 파일 경로
    batch_size : int, optional
        배치 크기 (기본값: 10000)

    Returns
    -------
    None
    """
    # GPU 사용 가능 여부 확인
    def check_gpu_support():
        try:
            from xgboost import config_context
            with config_context(verbosity=0):
                print("GPU is configured for use with XGBoost.")
        except ImportError:
            print("GPU support is not configured in XGBoost. Using CPU instead.")

    check_gpu_support()

    # 데이터 로드
    combined_data = pd.read_csv(input_csv)

    # Feature와 Label 분리
    X = combined_data.drop(["label", "KCNTOMS"], axis=1)  # KCNTOMS feature 제외
    y = combined_data["label"]

    # 레이블 인코딩
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # 범주형 및 수치형 Feature 분리
    categorical_features = [col for col in X.columns if set(X[col].unique()) <= {0, 1}]
    numerical_features = [col for col in X.columns if col not in categorical_features]

    # 스케일링 (수치형 데이터만)
    scaler = StandardScaler()
    scaler.fit(X[numerical_features])

    X_scaled = X.copy()
    X_scaled[numerical_features] = scaler.transform(X_scaled[numerical_features])

    # XGBoost 모델 학습 및 Feature 중요도 계산
    params = {
        'tree_method': 'hist',
        'device': 'cuda'  # GPU 사용
    }

    model = XGBClassifier(**params)
    model.fit(X_scaled, y_encoded)
    feature_importances = model.feature_importances_

    # Feature 중요도 저장
    importance_df = pd.DataFrame({"feature": X.columns, "importance": feature_importances})
    importance_df = importance_df.sort_values(by="importance", ascending=False)
    importance_df.to_csv(output_file, index=False, sep='\t')

    print(f"Feature importance saved to {output_file}")

if __name__ == "__main__":
    # 사용 예시
    input_csv = './src/consolidated_file_standardized.csv'
    output_file = './src/feature_importances.txt'
    calculate_feature_importances(input_csv, output_file)
