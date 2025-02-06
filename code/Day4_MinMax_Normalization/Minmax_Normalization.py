import os
import pandas as pd

para_list = ['KCNTOMS', 'KBCDO23', 'ZINST58']

def minmax(y, y_min, y_max):
    if y_min == y_max:
        return 0
    else:
        return (y - y_min) / (y_max - y_min)
    
# 폴더 경로 설정
folder_path = './Data'

# 모든 파일을 읽어서 하나의 데이터프레임으로 합침
all_data = pd.DataFrame()

for filename in os.listdir(folder_path):
    if filename.endswith('.csv'):
        file_path = os.path.join(folder_path, filename)
        # print(file_path)

        db = pd.read_csv(file_path)

        all_data = pd.concat([all_data, db], ignore_index=True)


# min, max 값 구해줍니다 =============================================================================
para_min = [all_data[para].min() for para in para_list]
para_max = [all_data[para].max() for para in para_list]
# print(para_min)
# print(para_max)

# min, max 값은 추후 사용을 위해 저장해줍시다 ==========================================================
with open('minmax.csv', 'w') as f:
    f.write(','.join(para_list) + '\n')
     # 두 번째 줄에 para_min 값을 작성
    f.write(','.join(map(str, para_min)) + '\n')
    # 세 번째 줄에 para_max 값을 작성
    f.write(','.join(map(str, para_max)) + '\n')


# 폴더 내에 있는 파일들에 대해 minmax 함수를 돌려줍시다 =================================================
# 저장할 폴더 경로 설정
new_path = './Data_minmax'

if not os.path.exists(new_path):
    os.makedirs(new_path)

for filename in os.listdir(folder_path):
    if filename.endswith('.csv'):
        file_path = os.path.join(folder_path, filename)

        # 정규화시킬 파일을 읽어들임
        db = pd.read_csv(file_path)

        # 새로운 파일 이름 설정
        new_filename = f'{filename[:-4]}_minmax.csv'  # filename만 사용하여 새로운 이름 생성
        new_file_path = os.path.join(new_path, new_filename)  # new_path에 저장할 경로 지정

        # 파일을 새 경로에 저장
        with open(new_file_path, 'w') as f:  # new_file_path 사용
            f.write(','.join(para_list) + '\n')

            for i in range(len(db)):
                out = []

                for j, para in enumerate(para_list):
                    out.append(minmax(db[para].iloc[i], para_min[j], para_max[j]))

                out = ','.join(map(str, out))
                f.write(f'{out}\n')

        print(f'{new_file_path} 저장 완료')
