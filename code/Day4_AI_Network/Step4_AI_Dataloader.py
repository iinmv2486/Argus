import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

class CustomDataset(Dataset):
    def __init__(self, file_path='', time_seq=1):
        # 디렉터리 존재 여부 확인
        if not os.path.isdir(file_path):
            raise ValueError(f"유효하지 않은 파일 경로: {file_path}")

        # 데이터 파일 경로
        file_list = [file for file in os.listdir(file_path) if file.endswith('.csv')]
        if len(file_list) == 0:
            raise ValueError(f"디렉터리에 CSV 파일이 없습니다: {file_path}")

        # 데이터와 레이블 초기화
        self.data = []
        self.label = []

        # 각 파일에서 데이터 로드 및 처리
        for file in file_list:
            df = pd.read_csv(os.path.join(file_path, file))
            
            if time_seq > 1:
                # 시퀀스 데이터를 생성
                for i in range(len(df) - time_seq):
                    self.data.append(df.iloc[i:i + time_seq, :-1].values)
                    self.label.append(df.iloc[i + time_seq - 1, -1])
            else:
                # 단일 시점 데이터를 처리
                for i in range(len(df)):
                    self.data.append(df.iloc[i, :-1].values)
                    self.label.append(df.iloc[i, -1])

        # 리스트를 numpy 배열로 변환
        self.data = np.array(self.data)
        self.label = np.array(self.label)

        # 데이터와 레이블 길이의 일관성 검사
        if len(self.data) != len(self.label):
            raise ValueError("데이터와 레이블의 길이가 일치하지 않습니다.")

        # numpy 배열을 PyTorch 텐서로 변환
        self.data = torch.tensor(self.data, dtype=torch.float32)
        self.label = torch.tensor(self.label, dtype=torch.long)

    def __len__(self):
        # 데이터의 총 샘플 수 반환
        return len(self.data)

    def __getitem__(self, index):
        # 주어진 인덱스의 데이터와 레이블 반환
        return self.data[index], self.label[index]

if __name__ == "__main__":
    # 사용 예제
    try:
        dataset = CustomDataset('./Day4_AI_Network/Data_minmax_labeling', time_seq=5)
        dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

        for batch_idx, (sample_data, labels) in enumerate(dataloader):
            print(f"배치 {batch_idx + 1}/{len(dataloader)}")
            print("샘플 데이터 크기:", sample_data.size())
            print("레이블 크기:", labels.size())
    except ValueError as e:
        print(f"오류: {e}")
