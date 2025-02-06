import numpy as np
import pandas as pd
from Filter_and_process_features import filter_and_process_features
from Split_dataset_files import split_dataset_files

def preprocess_data_with_sliding_window(file_path, segment_size=5, stride=1, label_col='label'):
    """
    CSV 내부 라벨 사용 + 슬라이딩 윈도우 방식으로 세그먼트화 예시
    1) CSV에서 라벨, 피처 분리
    2) stride 만큼씩 창(segment_size)을 옮기면서 세그먼트 생성
    3) 세그먼트 내 라벨 결정(예: 마지막 행 라벨)
    """

    df = pd.read_csv(file_path)
    
    row_labels = df[label_col].values
    feature_cols = [c for c in df.columns if c != label_col]
    row_data = df[feature_cols].values
    
    T, num_features = row_data.shape
    
    segments = []
    labels = []

    # 슬라이딩 윈도우: range(0, T-segment_size+1, stride)
    for start_idx in range(0, T - segment_size + 1, stride):
        end_idx = start_idx + segment_size
        
        segment_data = row_data[start_idx:end_idx, :]  # (segment_size, num_features)
        segment_label_vals = row_labels[start_idx:end_idx]  # 이 구간의 라벨
        
        # (예) 마지막 행 라벨
        segment_label = segment_label_vals[-1]
        
        segments.append(segment_data)
        labels.append(segment_label)
    
    segments = np.array(segments)
    labels = np.array(labels)
    
    return segments, labels



def load_segments_sliding_window(file_list, segment_size=5, stride=1, label_col='label'):
    all_segments, all_labels = [], []
    for file_path in file_list:
        segs, labs = preprocess_data_with_sliding_window(
            file_path, segment_size=segment_size, stride=stride, label_col=label_col
        )
        if segs.shape[0] == 0:
            # 이 파일에서 세그먼트가 0개면 건너뛰거나 예외 처리
            continue
        all_segments.append(segs)
        all_labels.append(labs)
    
    if len(all_segments) > 0:
        all_segments = np.concatenate(all_segments, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
    else:
        # 전부 0개일 수도 있으니 예외 처리
        all_segments = np.array([])
        all_labels = np.array([])
    
    return all_segments, all_labels

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

    # 형태 확인
    print("Train:", train_segments.shape, train_labels.shape)
    print("Val:  ", val_segments.shape,   val_labels.shape)
    print("Test: ", test_segments.shape,  test_labels.shape)