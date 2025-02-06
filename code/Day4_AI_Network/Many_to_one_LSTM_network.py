import torch
import torch.nn as nn
import torch.optim as optim

class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, num_layers=1):
        super(LSTMClassifier, self).__init__()
        # batch_first=True: 입력 텐서의 첫 번째 차원을 배치로 사용
        self.lstm = nn.LSTM(input_size=input_dim, 
                            hidden_size=hidden_dim, 
                            num_layers=num_layers, 
                            batch_first=True)
        
        # 최종 은닉 상태 -> 분류를 위한 fully-connected 레이어
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        """
        x: [배치 크기, 시점 길이, 입력 차원]
        """
        out, (h_n, c_n) = self.lstm(x)
        # h_n: [num_layers, 배치 크기, hidden_dim]
        last_hidden = h_n[-1]  # 가장 마지막 레이어의 최종 시점 은닉 상태
        logits = self.fc(last_hidden)  # [배치 크기, num_classes]
        return logits

    def save_network(self, path='AINetwork.pt'):
        """
        학습된 모델의 가중치를 파일로 저장합니다.
        """
        torch.save(self.state_dict(), path)
        print(f"네트워크가 '{path}'에 저장되었습니다.")

    def load_network(self, path='AINetwork.pt'):
        """
        저장된 모델의 가중치를 불러옵니다.
        """
        self.load_state_dict(torch.load(path))
        print(f"네트워크가 '{path}'에서 불러와졌습니다.")

    def net_training(self, training_data, training_target, epochs=10, lr=0.001):
        """
        간단한 학습 루프 예시입니다.
        training_data와 training_target은 텐서 형태(단일 배치 또는 미니배치)라고 가정합니다.
        epochs: 학습 에폭 수
        lr: 학습률
        """
        # 분류 문제이므로 CrossEntropyLoss 사용
        criterion = nn.CrossEntropyLoss()
        # Adam 옵티마이저 설정
        optimizer = optim.Adam(self.parameters(), lr=lr)

        # 학습 루프
        for epoch in range(epochs):
            # 기울기 초기화
            optimizer.zero_grad()

            # 순전파
            outputs = self.forward(training_data)
            
            # 손실 계산
            loss = criterion(outputs, training_target)
            
            # 역전파
            loss.backward()
            # 옵티마이저로 파라미터 갱신
            optimizer.step()

            # 간단한 학습 상황 출력
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

    def calculate_accuracy(self, network, data, target):
        """
        간단한 정확도(Accuracy)를 계산합니다.
        network: 학습된(혹은 학습 중) 모델 자체
        data: 입력 데이터 텐서 (배치 크기, 시점 길이, 입력 차원)
        target: 정답 라벨 (배치 크기)
        """
        with torch.no_grad():
            logits = network(data)             # [배치 크기, num_classes]
            predictions = torch.argmax(logits, dim=1)
            correct = (predictions == target).sum().item()
            accuracy = correct / data.size(0)

        return accuracy


