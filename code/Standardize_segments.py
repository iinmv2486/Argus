import numpy as np
from Filter_and_process_features import filter_and_process_features
from Split_dataset_files import split_dataset_files
from Segement_data import load_segments_sliding_window

def standardize_segments(train_segments, val_segments, test_segments):
    """
    데이터 세트를 표준화 (Standardization)합니다. 각 특성은 평균이 0, 표준편차가 1이 되도록 변환됩니다.

    Parameters
    ----------
    train_segments : numpy.ndarray
        학습 데이터 세트 (shape: num_samples x time_steps x num_features)
    val_segments : numpy.ndarray
        검증 데이터 세트 (shape: num_samples x time_steps x num_features)
    test_segments : numpy.ndarray
        테스트 데이터 세트 (shape: num_samples x time_steps x num_features)

    Returns
    -------
    tuple of numpy.ndarray
        표준화된 (train_segments, val_segments, test_segments)
    """
    # 학습 데이터로부터 평균과 표준편차 계산
    train_mean = train_segments.mean(axis=(0, 1))  # (num_features,)
    train_std = train_segments.std(axis=(0, 1))

    def standardize(data, data_mean, data_std):
        return (data - data_mean) / (data_std + 1e-8)

    # 각 데이터 세트를 표준화
    train_segments = standardize(train_segments, train_mean, train_std)
    val_segments = standardize(val_segments, train_mean, train_std)
    test_segments = standardize(test_segments, train_mean, train_std)

    return train_segments, val_segments, test_segments

if __name__ == "__main__":
    feature_importances_path = './대회_준비/Time_Series_Classification/nuclear_bootcamp/src/feature_importances.txt'
    base_path = "./대회_준비/Time_Series_Classification/nuclear_bootcamp/src"
    output_path = "./대회_준비/Time_Series_Classification/nuclear_bootcamp/Filtered_data"
    scenario_dict = filter_and_process_features(feature_importances_path, base_path, output_path)
    train_files, val_files, test_files = split_dataset_files(scenario_dict)

    segment_size = 5
    train_segments, train_labels = load_segments_sliding_window(train_files, segment_size=segment_size, label_col='label')
    val_segments,   val_labels   = load_segments_sliding_window(val_files,   segment_size=segment_size, label_col='label')
    test_segments,  test_labels  = load_segments_sliding_window(test_files,  segment_size=segment_size, label_col='label')
    
    train_segments, val_segments, test_segments = standardize_segments(train_segments, val_segments, test_segments)
