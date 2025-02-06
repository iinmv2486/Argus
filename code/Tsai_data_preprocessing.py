import numpy as np
from tsai.all import *
from Filter_and_process_features import filter_and_process_features
from Split_dataset_files import split_dataset_files
from Segement_data import load_segments_sliding_window
from Standardize_segments import standardize_segments

def preprocess_and_save_tsai_data(train_segments, train_labels, val_segments, val_labels, 
                                   test_segments, test_labels, save_path):
    """
    tsai 프레임워크에 맞게 데이터를 전처리하고 저장합니다.

    Parameters
    ----------
    train_segments : numpy.ndarray
        학습 데이터 세트 (shape: num_samples x seq_len x num_features)
    train_labels : numpy.ndarray
        학습 데이터 레이블
    val_segments : numpy.ndarray
        검증 데이터 세트 (shape: num_samples x seq_len x num_features)
    val_labels : numpy.ndarray
        검증 데이터 레이블
    test_segments : numpy.ndarray
        테스트 데이터 세트 (shape: num_samples x seq_len x num_features)
    test_labels : numpy.ndarray
        테스트 데이터 레이블
    save_path : str
        전처리된 데이터를 저장할 경로

    Returns
    -------
    None
        데이터가 저장됩니다.
    """
    # 데이터 형태 변환 (samples, seq_len, num_features) -> (samples, num_features, seq_len)
    train_X = train_segments.transpose(0, 2, 1).astype('float32')
    val_X = val_segments.transpose(0, 2, 1).astype('float32')
    test_X = test_segments.transpose(0, 2, 1).astype('float32')

    # 데이터 저장
    np.savez_compressed(save_path, train_X=train_X, train_y=train_labels, 
                        val_X=val_X, val_y=val_labels, 
                        test_X=test_X, test_y=test_labels)
    print(f"Preprocessed data saved to {save_path}")

def load_tsai_data_and_create_dls(save_path):
    """
    저장된 데이터를 로드하고 tsai DataLoaders를 생성합니다.

    Parameters
    ----------
    save_path : str
        저장된 데이터 파일 경로

    Returns
    -------
    TSDataLoaders
        tsai 데이터로더
    """
    # 데이터 로드
    data = np.load(save_path)
    train_X = data['train_X']
    train_y = data['train_y']
    val_X = data['val_X']
    val_y = data['val_y']

    # Train/Validation 데이터 결합
    train_size = len(train_X)
    val_size = len(val_X)

    X_2way = np.concatenate([train_X, val_X], axis=0)
    y_2way = np.concatenate([train_y, val_y], axis=0)

    train_idx = list(range(0, train_size))
    val_idx = list(range(train_size, train_size + val_size))

    splits = (train_idx, val_idx)

    # tsai Dataset 및 DataLoader 생성
    tfms = [None, [Categorize()]]
    dsets = TSDatasets(X_2way, y_2way, tfms=tfms, splits=splits)

    bs = 512
    dls = TSDataLoaders.from_dsets(dsets.train, dsets.valid, bs=[bs, bs * 2])

    print(f"DataLoaders created from {save_path}")
    return dls


if __name__ == "__main__":
    # 사용 예시
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

    save_path = "./대회_준비/Time_Series_Classification/nuclear_bootcamp/src/preprocessed_tsai_data.npz"
    preprocess_and_save_tsai_data(train_segments, train_labels, val_segments, val_labels, 
                                test_segments, test_labels, save_path)

    dls = load_tsai_data_and_create_dls(save_path)