import torch
import torch.nn as nn
import torch.optim as optim
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

# 데이터셋 및 데이터로더 설정
train_dataset = CustomDataset('./dataset/train', time_seq=5)
dataloader = DataLoader(train_dataset, batch_size=10, shuffle=True)

# 학습 파라미터 설정
epochs = 10
learning_rate = 0.001
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(Net_origin.parameters(), lr=learning_rate)

# 학습 루프
for epoch in range(epochs):
    Net_origin.train()  # 모델을 학습 모드로 설정
    epoch_loss = 0

    for index, (sample_data, labels) in enumerate(dataloader):
        # 데이터를 CUDA 장치로 이동
        sample_data = sample_data.to(device)
        labels = labels.to(device)

        # 옵티마이저 초기화
        optimizer.zero_grad()

        # 순전파
        outputs = Net_origin(sample_data)

        # 손실 계산
        loss = criterion(outputs, labels)

        # 역전파
        loss.backward()

        # 옵티마이저 업데이트
        optimizer.step()

        epoch_loss += loss.item()

    # 에포크 손실 출력
    print(f"Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss / len(dataloader):.4f}")

# 학습된 모델 저장
Net_origin.save_network("./models/Many_to_one_LSTM.pt")
