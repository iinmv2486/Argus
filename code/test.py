import torch
import torch.nn as nn
import torch.optim as optim

class ManyToOneLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
        super(ManyToOneLSTM, self).__init__()
        
        # LSTM Layer
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        
        # Fully Connected Layer to output predictions
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x: (batch_size, sequence_length, input_dim)
        
        # LSTM output
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # We take the last hidden state for many-to-one output
        last_hidden = hidden[-1]  # (batch_size, hidden_dim)
        
        # Fully connected layer
        output = self.fc(last_hidden)  # (batch_size, output_dim)
        
        return output

# Example Usage
def train_model():
    # Hyperparameters
    input_dim = 1         # Feature dimension
    hidden_dim = 50       # Hidden state size
    output_dim = 1        # Output dimension
    num_layers = 2        # Number of LSTM layers
    learning_rate = 0.001
    num_epochs = 50

    # Model, Loss, Optimizer
    model = ManyToOneLSTM(input_dim, hidden_dim, output_dim, num_layers)
    criterion = nn.MSELoss()  # Loss for regression
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Example Data (sine wave for simplicity)
    sequence_length = 10
    x_train = torch.linspace(0, 100, steps=500).unsqueeze(1)  # (500, 1)
    y_train = torch.sin(x_train)  # Target values

    # Preparing data in sequences
    def create_sequences(data, seq_length):
        sequences = []
        targets = []
        for i in range(len(data) - seq_length):
            seq = data[i:i+seq_length]
            target = data[i+seq_length]
            sequences.append(seq)
            targets.append(target)
        return torch.stack(sequences), torch.stack(targets)

    x_train_seq, y_train_seq = create_sequences(y_train, sequence_length)

    # Training Loop
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        # Forward pass
        outputs = model(x_train_seq)  # (batch_size, output_dim)
        loss = criterion(outputs, y_train_seq)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        if (epoch+1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

    print("Training complete.")

if __name__ == "__main__":
    train_model()
