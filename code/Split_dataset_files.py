import random
from Filter_and_process_features import filter_and_process_features

def split_dataset_files(scenario_dict, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2):
    """
    시나리오 파일들을 Train/Validation/Test 세트로 분할합니다.

    Parameters
    ----------
    scenario_dict : dict
        라벨별 파일 경로 리스트를 포함하는 딕셔너리
    train_ratio : float, optional
        Train 세트 비율 (기본값: 0.7)
    val_ratio : float, optional
        Validation 세트 비율 (기본값: 0.1)
    test_ratio : float, optional
        Test 세트 비율 (기본값: 0.2)

    Returns
    -------
    tuple of lists
        (train_files, val_files, test_files): 각 세트에 포함된 파일 경로 리스트
    """
    train_files, val_files, test_files = [], [], []

    for label, file_list in scenario_dict.items():
        total_files = len(file_list)
        random.shuffle(file_list)

        n_train = int(total_files * train_ratio)
        n_val = int(total_files * val_ratio)
        
        train_part = file_list[:n_train]
        val_part = file_list[n_train:n_train + n_val]
        test_part = file_list[n_train + n_val:]

        train_files.extend(train_part)
        val_files.extend(val_part)
        test_files.extend(test_part)

    print(f"Train files: {len(train_files)}, Val files: {len(val_files)}, Test files: {len(test_files)}")
    return train_files, val_files, test_files


if __name__ == "__main__":
    feature_importances_path = './대회_준비/Time_Series_Classification/nuclear_bootcamp/src/feature_importances.txt'
    base_path = "./대회_준비/Time_Series_Classification/nuclear_bootcamp/src"
    output_path = "./대회_준비/Time_Series_Classification/nuclear_bootcamp/Filtered_data"
    scenario_dict = filter_and_process_features(feature_importances_path, base_path, output_path)
    train_files, val_files, test_files = split_dataset_files(scenario_dict)
