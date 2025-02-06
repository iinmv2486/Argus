import torch
from torch.utils.data import DataLoader
from Step4_AI_Dataloader import CustomDataset
from Many_to_one_LSTM_network import LSTMClassifier

# 모델 초기화에 필요한 파라미터 설정
input_dim = 72        # 각 시점에서의 입력 피처 수
hidden_dim = 50       # LSTM의 은닉 상태 차원
num_classes = 5       # 분류 클래스 수
num_layers = 2        # LSTM 레이어 수

# CUDA 장치 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"사용 중인 장치: {device}")

# LSTMClassifier 인스턴스 생성 및 장치로 이동
Net_origin = LSTMClassifier(input_dim=input_dim, hidden_dim=hidden_dim, num_classes=num_classes, num_layers=num_layers)
Net_origin.to(device)

# 저장된 네트워크 가중치 로드
Net_origin.load_network('./models/Many_to_one_LSTM.pt')

# 테스트 데이터셋 및 데이터로더 설정
test_dataset = CustomDataset('./dataset/test', time_seq=5)
test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)

def evaluate_accuracy(model, data_loader, device):
    model.eval()  # 모델을 평가 모드로 설정
    correct = 0
    total = 0

    with torch.no_grad():  # 평가 중에는 기울기 계산 비활성화
        for sample_data, labels in data_loader:
            # 데이터를 장치로 이동
            sample_data, labels = sample_data.to(device), labels.to(device)

            # 모델 출력 계산
            outputs = model(sample_data)

            # 예측값 계산
            _, predicted = torch.max(outputs, 1)  # 각 샘플의 최대값 인덱스를 가져옴

            # 정확도 계산
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total * 100
    return accuracy

# 테스트 데이터로 정확도 평가
accuracy = evaluate_accuracy(Net_origin, test_loader, device)
print(f"테스트 데이터 정확도: {accuracy:.2f}%")
