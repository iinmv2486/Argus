import os
import pandas as pd
import json

def filter_and_standardize_features(
    feature_importances_path,
    scaler_parameters_path,
    base_path,
    output_path,
    importance_threshold=0.0001,
):
    """
    Feature Importance와 표준화 파라미터를 기반으로 데이터를 필터링 및 전처리하여 저장합니다.

    Parameters
    ----------
    feature_importances_path : str
        Feature 중요도 파일 경로 (feature_importances.txt 형식)
    scaler_parameters_path : str
        표준화 파라미터 파일 경로 (scaler_parameters.json 형식)
    base_path : str
        원본 데이터가 포함된 상위 폴더 경로
    output_path : str
        전처리된 데이터를 저장할 상위 폴더 경로
    importance_threshold : float, optional
        Feature 중요도의 임계값 (기본값: 0.0001)

    Returns
    -------
    dict
        전처리된 파일 경로를 포함하는 시나리오 딕셔너리
    """
    # Feature Importance 로드
    feature_importances = pd.read_csv(feature_importances_path, sep="\t")
    selected_features = feature_importances[
        feature_importances['importance'] > importance_threshold
    ]['feature'].tolist()

    if not selected_features:
        raise ValueError("선택된 feature가 없습니다. 중요도 임계값을 확인하세요.")

    print("선택된 features:", selected_features)

    # 표준화 파라미터 로드
    try:
        with open(scaler_parameters_path, 'r') as f:
            scaler_parameters = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError("표준화 파라미터 파일을 찾을 수 없습니다. 경로를 확인하세요.")

    mean = scaler_parameters['mean']
    std = scaler_parameters['std']

    # 출력 폴더 생성
    os.makedirs(output_path, exist_ok=True)

    # 시나리오 딕셔너리 및 진행 상태 변수 초기화
    scenario_dict = {}
    total_files = 0
    processed_files = 0

    for label in range(5):  # 폴더 이름이 0, 1, 2, 3, 4로 분류되어 있다고 가정
        folder_path = os.path.join(base_path, str(label))
        output_folder = os.path.join(output_path, str(label))
        os.makedirs(output_folder, exist_ok=True)

        if os.path.exists(folder_path):
            csv_files = [
                os.path.join(folder_path, file)
                for file in os.listdir(folder_path)
                if file.endswith(".csv")
            ]
            scenario_dict[label] = []
            total_files += len(csv_files)

            for i, csv_file in enumerate(csv_files, 1):
                try:
                    print(f"Processing file {i}/{len(csv_files)} in folder {label}...")

                    # CSV 파일 로드 및 Feature 필터링
                    data = pd.read_csv(csv_file)

                    # label 칼럼 유지
                    if 'label' not in data.columns:
                        raise ValueError(f"'{csv_file}'에 'label' 칼럼이 없습니다.")

                    columns_to_keep = [
                        col for col in selected_features if col in data.columns
                    ] + ['label']

                    if len(columns_to_keep) == 1:  # label 칼럼만 있는 경우
                        raise ValueError("필터링된 데이터에 사용할 feature가 없습니다.")

                    filtered_data = data[columns_to_keep].copy()

                    # 표준화 수행 (label 칼럼 제외)
                    for col in selected_features:
                        if col in mean and col in std and col in filtered_data.columns:
                            filtered_data[col] = (filtered_data[col] - mean[col]) / std[col]

                    # 전처리된 데이터 저장
                    processed_file_path = os.path.join(output_folder, os.path.basename(csv_file))
                    filtered_data.to_csv(processed_file_path, index=False)

                    scenario_dict[label].append(processed_file_path)
                    processed_files += 1
                except Exception as e:
                    print(f"Error processing file {csv_file}: {e}")
        else:
            scenario_dict[label] = []

    # 결과 출력
    print("Processed Scenario Dictionary (파일 목록):")
    for label, files in scenario_dict.items():
        print(f"Folder label {label}: {len(files)} files")

    if total_files > 0:
        processed_percentage = (processed_files / total_files) * 100
        print(f"총 {total_files}개의 파일 중 {processed_files}개의 파일이 처리되었습니다. (처리 비율: {processed_percentage:.2f}%)")
    else:
        print("처리할 파일이 없습니다.")

    return scenario_dict


if __name__ == "__main__":
    # 사용 예시
    feature_importances_path = './src/feature_importances.txt'
    scaler_parameters_path = './src/scaler_parameters.json'
    base_path = "./src"
    output_path = "./Filtered_Data"

    scenario_dict = filter_and_standardize_features(
        feature_importances_path,
        scaler_parameters_path,
        base_path,
        output_path
    )
