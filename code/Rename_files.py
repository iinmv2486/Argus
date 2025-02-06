import os

def rename_files_in_subfolders(base_folder, subfolders=None):
    """
    주어진 base_folder 아래 지정된 subfolders(기본값: ["0","1","2","3","4"])를 순회하며,
    파일명을 '정상적인' 형태로 바꿔주는 함수다.
    예: ".csv.csv"가 붙어 있으면 ".csv"로 교체하여 재명명한다.

    Parameters
    ----------
    base_folder : str
        상위 폴더 경로
    subfolders : list[str], optional
        순회할 하위 폴더 목록 (기본값: ["0", "1", "2", "3", "4"])
    """

    if subfolders is None:
        subfolders = ["0", "1", "2", "3", "4"]

    for folder_name in subfolders:
        folder_path = os.path.join(base_folder, folder_name)

        # 해당 폴더가 없으면 스킵
        if not os.path.exists(folder_path):
            print(f"Folder '{folder_name}' does not exist. Skipping.")
            continue

        # 폴더 내 모든 파일을 순회
        for file_name in os.listdir(folder_path):
            old_file_path = os.path.join(folder_path, file_name)

            # 파일이 아닌 경우(디렉토리 등) 스킵
            if not os.path.isfile(old_file_path):
                continue

            # 1) 확장자 중복(.csv.csv)을 단일 .csv로 바꾸는 예시
            #    만약 "파일명.csv.csv" 형태인 경우에만 변경
            if file_name.endswith(".csv.csv"):
                new_file_name = file_name[:-4]  # 뒤쪽 ".csv" 하나만 잘라냄
                new_file_path = os.path.join(folder_path, new_file_name)
                try:
                    os.rename(old_file_path, new_file_path)
                    print(f"Renamed: {file_name} -> {new_file_name}")
                except Exception as e:
                    print(f"Failed to rename {old_file_path}: {e}")


if __name__ == "__main__":
    base_folder_path = "./src"  # "src" 예시
    rename_files_in_subfolders(base_folder_path)