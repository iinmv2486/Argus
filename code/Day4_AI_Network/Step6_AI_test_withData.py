import torch.nn as nn
import torch
from torch.nn.functional import softmax, relu
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from Step4_AI_Dataloader import CustomDataset
from Many_to_one_LSTM_network import LSTMClassifier
from CNSenv import CommunicatorCNS

# 모델 초기화에 필요한 파라미터 설정
input_dim = 72        # 각 시점에서의 입력 피처 수
hidden_dim = 50       # LSTM의 은닉 상태 차원
num_classes = 5       # 분류 클래스 수
num_layers = 2        # LSTM 레이어 수


def minmax(y, y_min, y_max):
    if y_min == y_max:
        return 0
    else:
        return (y - y_min) / (y_max - y_min)

if __name__ == "__main__":
    Net_origin = LSTMClassifier(input_dim=input_dim, hidden_dim=hidden_dim, num_classes=num_classes, num_layers=num_layers)
    Net_origin.load_network('Many_to_one_LSTM.pt')

    cnsenv = CommunicatorCNS(com_ip= '172.30.1.75', com_port=7102)

    para_list = ['KCNTOMS', 'KBCDO23', 'ZINST58']      
    para_min = [0, 7, 101.3265609741211]               
    para_max = [52, 99, 156.2834014892578]         


    #=============================================================
    while True:
        is_updated = cnsenv.read_data()

        if is_updated:
            db_line = []
            for j, para in enumerate(para_list):
                db_line.append(minmax(cnsenv.mem[para]['Val'], para_min[j], para_max[j]))

            print(f'db_line : {db_line}')

            # AI Network -----------------------------------------

            inputs = torch.tensor([db_line], dtype=torch.float32)

            out = Net_orgin.forward(inputs)

            out = torch.argmax(softmax(out, dim=1)).tolist()
            
            print(f'Net Prediect : {out}')

        else:
            pass
    #=============================================================