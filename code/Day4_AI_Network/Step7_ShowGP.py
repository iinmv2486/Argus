from CNSenv import CommunicatorCNS
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque
import torch
from torch.nn.functional import softmax
import numpy as np
from Many_to_one_LSTM_network import LSTMClassifier
import pandas as pd
import json



# 🔹 시퀀스를 쌓는 Buffer 클래스 추가
class SequenceBuffer:
    def __init__(self, sequence_length):
        """
        시퀀스를 원하는 길이만큼 모아서 반환하는 버퍼 클래스
        :param sequence_length: AI 모델에 입력으로 보낼 시퀀스 길이
        """
        self.sequence_length = sequence_length
        self.buffer = deque(maxlen=sequence_length)  # 최대 길이를 제한하여 오래된 데이터 자동 삭제

    def add_data(self, new_data):
        """
        새로운 데이터를 추가하고, 시퀀스가 충분하면 반환
        :param new_data: 새로운 입력 데이터 리스트
        :return: 시퀀스 길이가 충분하면 리스트 반환, 부족하면 None 반환
        """
        self.buffer.append(new_data)  # 데이터 추가
        
        if len(self.buffer) == self.sequence_length:  # 시퀀스 길이가 충족되면 반환
            return list(self.buffer)  # deque → list 변환 후 반환
        
        return None  # 아직 데이터가 부족하면 None 반환

if __name__ == "__main__":
    
    def standardize(y, mean, std):
        """Standardization Function"""
        try:
            if std == 0:
                raise ValueError("Standard deviation cannot be zero for standardization.")
            standardized_value = (y - mean) / std
            return standardized_value
        except Exception as e:
            print(f"Error during standardization: {e}")
            return None


    # CNS 환경 설정
    cnsenv = CommunicatorCNS(com_ip='192.168.0.10', com_port=7132)

    feature_importances_path = './src/feature_importances.txt'
    feature_importances = pd.read_csv(feature_importances_path, sep="\t")

    para_list = ['H2CONC', 'VSUMP', 'DSECON', 'ZINST63', 'UFUEL18', 'CBSUMP', 'BLV2', 'XPMOUT', 'CBINTR', 'ZINST28', 'UNORDT', 'XPMBAL', 'BHV51', 'WLETDNO', 'QCSLOC', 'DRADRE', 'ZINST64', 'TCIRCD', 'BPV145I', 'WRHSTM', 'TCVLOC', 'KFIRST', 'ZCVLP', 'UAVLEG1', 'ZINST54', 'CAUTRDC', 'ZINST61', 'UCOOL', 'WCPLN1', 'WRHDRNA', 'WHPRH', 'ZVCT', 'FEICDP', 'ZINST51', 'UCWOUT', 'UCTMT', 'HHPRH', 'PHPTIN', 'ZINST102', 'UDHOCO2', 'WHPHBYA', 'BFWMAI', 'DECH', 'WAFWTK', 'WCHARGT', 'PCNDS', 'HRHTR', 'FHTRDP', 'KZROD25', 'UCNDTK', 'ZINST36', 'ZINST105', 'ZSGNOR2', 'ULPHCD', 'XSEBAL', 'ZINST78', 'UCOLEG1', 'UFWLN1', 'ZSGN3', 'DCTMT', 'URHTUB', 'UCOLEGM', 'BPV145', 'WFDW', 'PCDP', 'CBRWST', 'BFV489', 'WNETCH', 'CBVCT', 'UFUEL17', 'UCCWSW', 'WCSLOC']
    
    scaler_parameters_path = './src/scaler_parameters.json'

    # 표준화 파라미터 로드
    try:
        with open(scaler_parameters_path, 'r') as f:
            scaler_parameters = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError("표준화 파라미터 파일을 찾을 수 없습니다. 경로를 확인하세요.")

    mean = scaler_parameters['mean']
    std = scaler_parameters['std']
    
    # 딕셔너리를 리스트로 변환
    para_mean = np.array(list(mean.values()))
    para_std = np.array(list(std.values()))

    print("Hello", para_mean.shape, para_std.shape)

    # AI 네트워크 로드
    # 모델 초기화에 필요한 파라미터 설정
    input_dim = 72        # 각 시점에서의 입력 피처 수
    hidden_dim = 50       # LSTM의 은닉 상태 차원
    num_classes = 5       # 분류 클래스 수
    num_layers = 2        # LSTM 레이어 수

    # CUDA 장치 설정
    Net_origin = LSTMClassifier(input_dim=input_dim, hidden_dim=hidden_dim, num_classes=num_classes, num_layers=num_layers)
    Net_origin.load_network('./models/Many_to_one_LSTM.pt')

    # 데이터 저장 변수
    sim_time = []
    net_out = { "NORMAL": [], "LOCA": [], "SGTR": [], "MSLB_in": [], "MSLB_out": [] }
    NetDiag = ''

    # 시퀀스 길이 설정 및 버퍼 생성
    sequence_length = 5
    buffer = SequenceBuffer(sequence_length)

    # 그래프 초기화
    fig, ax = plt.subplots(figsize=(8,6))
    ax.set_position([0.10, 0.30, 0.85, 0.65])

    # 그래프 선 생성
    colors = {'NORMAL': 'black', 'LOCA': 'blue', 'SGTR': 'red', 'MSLB_in': 'green', 'MSLB_out': 'magenta'}
    lines = {key: ax.plot([], [], label=key, color=colors[key], linewidth=2.5)[0] for key in net_out}

    # 레이블 생성
    Label_Dig = fig.text(0.1, 0.15, f'Diagnosis : {NetDiag}', wrap=True)
    Labels = {
        "NORMAL": fig.text(0.1, 0.10, '', wrap=True, backgroundcolor='#eefade'),
        "LOCA": fig.text(0.1, 0.05, '', wrap=True, backgroundcolor='#eefade'),
        "SGTR": fig.text(0.1, 0.01, '', wrap=True, backgroundcolor='#eefade'),
        "MSLB_in": fig.text(0.5, 0.10, '', wrap=True, backgroundcolor='#eefade'),
        "MSLB_out": fig.text(0.5, 0.05, '', wrap=True, backgroundcolor='#eefade')
    }

    ax.set_title("AI Result")
    ax.set_xlabel("Time (sim. tick)")
    ax.set_ylabel("SoftMax Out")
    ax.axhline(y=0.9, linestyle=':', color='black')
    ax.legend(loc='upper center', ncol=5, bbox_to_anchor=(0.5, 1))

    with open('GP_Result_team.csv', 'w') as f:
        f.write(f'Tick,NORMAL,LOCA,SGTR,MSLB_in,MSLB_out,Diagnosis,Diagnosis_label\n')

    def update(frame):
        """실시간 업데이트"""
        is_updated = cnsenv.read_data()

        if is_updated:
            db_line = np.array([cnsenv.mem[para]['Val'] for j, para in enumerate(para_list)])
            print("line", db_line.shape)
            # db_line_normalized = standardize(db_line,para_mean,para_std)
            db_line_normalized  = (db_line - para_mean) / para_std
            print("norm", db_line_normalized)
            # 시퀀스 버퍼에 데이터 추가
            sequence = buffer.add_data(db_line_normalized)
            print(np.array(sequence).shape)

            if sequence:
                print(f"데이터 {sequence_length}개가 쌓였음! AI 모델에 전달할 시퀀스:")

                # Torch Tensor 변환 (batch 차원 추가)
                inputs = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0)  # (1, seq_length, input_size)
                print(inputs.size())

                # 모델 예측
                with torch.no_grad():  # 그래디언트 계산 방지
                    out = Net_origin.forward(inputs)

                print(out)
                # Softmax 적용 후 argmax로 가장 확률 높은 클래스 선택
                soft_out = softmax(out, dim=1).tolist()[0]  # 확률값 리스트 [5개]
                pred_idx = torch.argmax(torch.tensor(soft_out)).item()
                NetDiag = ["NORMAL", "LOCA", "SGTR", "MSLB inside", "MSLB outside"][pred_idx]

                soft_out_w = list(soft_out)
                soft_out_w.insert(0,str(cnsenv.mem["KCNTOMS"]["Val"]))
                soft_out_w = ','.join(map(str,soft_out_w))
                with open('GP_Result_team.csv','a') as f:
                    f.write(f'{soft_out_w},{str(pred_idx)},{str(NetDiag)}\n')

                # CNS 시뮬레이션 시간
                current_time = cnsenv.mem['KCNTOMS']['Val']
                sim_time.append(current_time)

                # 확률 저장
                for i, key in enumerate(net_out.keys()):
                    net_out[key].append(soft_out[i])

                # 그래프 데이터 업데이트
                for key, line in lines.items():
                    line.set_data(sim_time, net_out[key])

                # 레이블 업데이트
                Label_Dig.set_text(f'Diagnosis : {NetDiag}')
                for i, key in enumerate(Labels.keys()):
                    print(i, key, net_out[key][-1] * 100)
                    Labels[key].set_text(f'{key} : {net_out[key][-1] * 100:.2f}%')

                # 그래프 스케일 조정
                ax.relim()
                ax.autoscale_view() 

            else:
                remaining = sequence_length - len(buffer.buffer)  # 남은 데이터 개수 확인
                print(f"데이터가 부족함. {remaining}개 더 필요함...")

    # 애니메이션 실행
    ani = FuncAnimation(fig, update, interval=100, cache_frame_data=False)
    plt.show()
