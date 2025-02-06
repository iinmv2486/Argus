import os
import pandas as pd
from tqdm import tqdm

def check_null_values_in_files(base_folder, subfolders=None, zero_var_file=None):
    """
    주어진 base_folder 내의 subfolders 목록(기본값: ["0","1","2","3","4"])을 순회하면서,
    각 폴더에 있는 모든 CSV 파일에서 결측값(None/NaN)이 발생하는 열과 그 개수를 확인합니다.

    Parameters
    ----------
    base_folder : str
        상위 폴더 경로 (예: "src")
    subfolders : list of str, optional
        확인할 하위 폴더 리스트 (기본값: ["0","1","2","3","4"])
    zero_var_file : str, optional
        제거할 열 이름이 포함된 파일 경로 (기본값: None)

    Returns
    -------
    dict
        {
            "폴더명/파일명.csv": {
                "열이름": 결측치 개수,
                ...
            },
            ...
        }
        형태의 딕셔너리.
        결측치가 없는 파일은 결과에 포함되지 않습니다.
    """

    if subfolders is None:
        subfolders = ["0", "1", "2", "3", "4"]

    # zero_var_file에서 열 이름 불러오기
    zero_var_cols = []
    if zero_var_file:
        try:
            with open(zero_var_file, 'r') as f:
                zero_var_cols = eval(f.read())
        except Exception as e:
            print(f"Error reading zero variance columns file: {e}")

    # 결과를 저장할 딕셔너리
    null_info = {}

    # 전체 파일 수 계산
    total_files = 0
    for folder in subfolders:
        folder_path = os.path.join(base_folder, folder)
        if os.path.exists(folder_path):
            total_files += sum(
                1
                for fname in os.listdir(folder_path)
                if fname.endswith(".csv")
            )

    with tqdm(total=total_files, desc="Checking null values", unit="file") as pbar:
        for folder_name in subfolders:
            folder_path = os.path.join(base_folder, folder_name)
            if not os.path.exists(folder_path):
                print(f"Folder '{folder_name}' does not exist. Skipping.")
                continue

            for file_name in os.listdir(folder_path):
                if file_name.endswith(".csv"):
                    file_path = os.path.join(folder_path, file_name)

                    # CSV 파일 로딩
                    try:
                        data = pd.read_csv(file_path)

                        # zero_var_cols 제거
                        data.drop(zero_var_cols, axis=1, errors='ignore', inplace=True)

                        # 결측치 체크
                        none_counts = data.isnull().sum()
                        none_counts = none_counts[none_counts > 0]

                        if not none_counts.empty:
                            # 폴더/파일명을 key로 하여 어떤 열에 어느 정도 null이 있는지 저장
                            file_key = os.path.join(folder_name, file_name)
                            null_info[file_key] = none_counts.to_dict()

                    except Exception as e:
                        print(f"Error reading file {file_path}: {e}")

                    pbar.update(1)

    # 결과 요약 출력
    if not null_info:
        print("No files with None/NaN values found in the given folders.")
    else:
        print("\n=== Files containing None or NaN values ===")
        for file_key, null_columns in null_info.items():
            print(f"\n[ {file_key} ]")
            for col_name, count in null_columns.items():
                print(f"  - {col_name}: {count} 개")

    return null_info

if __name__ == "__main__":
    base_folder = "./src"
    zero_var_file = "./src/zero_variance_list.txt"

    result = check_null_values_in_files(base_folder, zero_var_file=zero_var_file)
