import os
import pandas as pd
from tqdm import tqdm

# 모든 데이터를 하나의 CSV 파일로 통합하여 저장하거나 기존 파일 로드
def consolidate_data(base_folder, zero_var_file=None):
    """
    주어진 base_folder 내에 "0", "1", "2", "3", "4" 폴더가 있고,
    각 폴더 안의 CSV 파일을 하나로 병합하여 consolidated_file.csv로 저장하는 함수입니다.
    zero_var_cols가 주어지면, 해당 칼럼들은 병합 시 제거합니다.

    Parameters
    ----------
    base_folder : str
        데이터가 들어있는 상위 폴더 경로
    zero_var_cols : list, optional
        분산이 0인 칼럼 이름의 리스트 (기본값: None)
    """

    output_file = os.path.join(base_folder, "consolidated_file.csv")

    # 이미 병합된 파일이 있다면, 그대로 반환
    if os.path.exists(output_file):
        print(f"consolidated_file.csv found at {output_file}. Loading directly.")
        return output_file

    all_data = []
    folders = ["0", "1", "2", "3", "4"]

    # zero_var_file에서 열 이름 불러오기
    zero_var_cols = []
    if zero_var_file:
        try:
            with open(zero_var_file, 'r') as f:
                zero_var_cols = eval(f.read())
        except Exception as e:
            print(f"Error reading zero variance columns file: {e}")
    
    # 전체 파일 수 계산
    total_files = sum(
        len(os.listdir(os.path.join(base_folder, folder))) 
        for folder in folders 
        if os.path.exists(os.path.join(base_folder, folder))
    )

    processed_files = 0

    with tqdm(total=total_files, desc="Processing files", unit="file") as pbar:
        for folder_name in folders:
            folder_path = os.path.join(base_folder, folder_name)
            if not os.path.exists(folder_path):  # 해당 폴더가 없으면 스킵
                print(f"Folder {folder_name} does not exist. Skipping.")
                continue

            for file_name in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file_name)
                if file_name.endswith(".csv"):  # CSV 파일만 처리
                    try:
                        # CSV 파일 로딩
                        data = pd.read_csv(file_path)

                        # 분산 0인 칼럼을 제거
                        data.drop(zero_var_cols, axis=1, inplace=True, errors='ignore')

                        # 최종 병합 리스트에 추가
                        all_data.append(data)
                    except Exception as e:
                        print(f"Error processing file {file_path}: {e}")

                processed_files += 1
                pbar.update(1)  # 진행 상황 업데이트

    # 하나의 데이터프레임으로 결합 및 저장
    if all_data:
        combined_data = pd.concat(all_data, ignore_index=True)
        combined_data.to_csv(output_file, index=False)
        print(f"All data consolidated into {output_file}")
    else:
        print("No data was processed. Check your files and folders.")

    return output_file


# 데이터 통합 또는 기존 파일 로드
if __name__ == "__main__":
    base_folder = "./src"
    zero_var_file = "./src/zero_variance_list.txt"

    result = consolidate_data(base_folder, zero_var_file=zero_var_file)
