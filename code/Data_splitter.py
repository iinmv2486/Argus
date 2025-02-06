import os
import shutil
import random

def split_dataset(src_root, dst_root, train_ratio=0.8, valid_ratio=0.1, test_ratio=0.1):
    """
    src_root : 원본 데이터가 들어 있는 최상위 폴더 경로 (예: "src")
    dst_root : 새롭게 데이터를 분배할 폴더 경로 (예: "dataset")
    train_ratio, valid_ratio, test_ratio : 데이터 분할 비율
    """

    # 비율 합계가 1이 맞는지 확인
    if not abs(train_ratio + valid_ratio + test_ratio - 1.0) < 1e-9:
        raise ValueError("train_ratio + valid_ratio + test_ratio의 합은 1이어야 합니다.")

    # 결과를 저장할 dst_root 폴더 생성
    for split in ['train', 'valid', 'test']:
        split_path = os.path.join(dst_root, split)
        os.makedirs(split_path, exist_ok=True)

    total_files = 0
    processed_files = 0

    for label in range(5):  # 폴더 이름이 0, 1, 2, 3, 4로 분류되어 있다고 가정
        folder_path = os.path.join(src_root, str(label))

        if os.path.exists(folder_path):
            csv_files = [
                os.path.join(folder_path, file)
                for file in os.listdir(folder_path)
                if file.endswith(".csv")
            ]
            total_files += len(csv_files)

            # 파일 리스트 무작위 셔플
            random.shuffle(csv_files)

            # 학습, 검증, 테스트 분할 인덱스 계산
            train_count = int(len(csv_files) * train_ratio)
            valid_count = int(len(csv_files) * valid_ratio)

            train_files = csv_files[:train_count]
            valid_files = csv_files[train_count:train_count + valid_count]
            test_files = csv_files[train_count + valid_count:]

            # 파일 복사 함수 정의
            def copy_files(file_list, split):
                for f in file_list:
                    src_file_path = f
                    dst_file_path = os.path.join(dst_root, split, os.path.basename(f))
                    shutil.copy2(src_file_path, dst_file_path)

            # 파일 복사
            copy_files(train_files, 'train')
            copy_files(valid_files, 'valid')
            copy_files(test_files, 'test')

            processed_files += len(train_files) + len(valid_files) + len(test_files)

    # 결과 출력
    print(f"총 {total_files}개의 파일 중 {processed_files}개의 파일이 처리되었습니다.")

if __name__ == "__main__":
    # src 폴더와 새로 분배될 폴더(dataset)를 설정
    src_root = "Filtered_data"
    dst_root = "dataset"

    # 원하는 비율 설정(예: 80%, 10%, 10%)
    train_ratio = 0.7
    valid_ratio = 0.1
    test_ratio = 0.2

    # 함수 실행
    split_dataset(
        src_root=src_root,
        dst_root=dst_root,
        train_ratio=train_ratio,
        valid_ratio=valid_ratio,
        test_ratio=test_ratio
    )
