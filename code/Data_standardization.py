import pandas as pd
from sklearn.preprocessing import StandardScaler
import json

# 중요도가 높은 feature들만 선택
feature_importances_path = './src/feature_importances.txt'  # Feature 중요도 파일 경로
importance_threshold = 0.0001

# 중요도 파일 로드
try:
    feature_importances = pd.read_csv(feature_importances_path, sep="\t")
    selected_features = feature_importances[
        feature_importances['importance'] > importance_threshold
    ]['feature'].tolist()
    if not selected_features:
        raise ValueError("선택된 feature가 없습니다. 중요도 임계값을 확인하세요.")
    print("선택된 features:", selected_features)
except FileNotFoundError:
    raise FileNotFoundError("Feature 중요도 파일을 찾을 수 없습니다. 경로를 확인하세요.")

# 데이터 로드
file_path = './src/consolidated_file.csv'  # 통합 파일 경로
try:
    data = pd.read_csv(file_path)
    print("데이터가 성공적으로 로드되었습니다.")
except FileNotFoundError:
    raise FileNotFoundError("지정한 파일을 찾을 수 없습니다. 파일 경로를 확인하세요.")

# 중요도가 높은 feature들로 필터링
filtered_data = data[selected_features].copy()

# 표준화 수행
scaler = StandardScaler()
data_standardized = pd.DataFrame(scaler.fit_transform(filtered_data), columns=selected_features)

# mean과 std 값을 저장
scaler_parameters = {
    "mean": dict(zip(selected_features, scaler.mean_)),
    "std": dict(zip(selected_features, scaler.scale_))
}
params_output_path = './src/scaler_parameters.json'
with open(params_output_path, 'w') as f:
    json.dump(scaler_parameters, f, indent=4)

print(f"표준화 과정에서 사용된 mean과 std 값이 {params_output_path}에 저장되었습니다.")

# 표준화된 데이터를 기존 데이터에 통합
data_standardized_full = data.copy()
data_standardized_full[selected_features] = data_standardized

# 표준화된 데이터를 새 파일로 저장
output_path = './src/consolidated_file_standardized.csv'
data_standardized_full.to_csv(output_path, index=False)
print(f"표준화된 데이터가 {output_path}에 저장되었습니다.")
