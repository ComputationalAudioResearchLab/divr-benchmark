from .Base import Base, PytorchBase
from .svm import SVM, SGDSVM


class Hybrid(PytorchBase):
    def __init__(self, input_shape):
        super(HybridModel, self).__init__()
        
        # Convolutional Blocks
        self.conv_layers = nn.Sequential(
            *[nn.Sequential(
                nn.Conv1d(input_shape if i == 0 else 64, 64, 3),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.MaxPool1d(2)
            ) for i in range(5)]
        )
        
        # RNN Block
        self.lstm = nn.LSTM(64, 64, batch_first=True)
        
        # Dense Layers
        self.fc1 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, 2) # 2 classes
    
    def forward(self, x):
        x = self.conv_layers(x)  # Output from Conv layers
        x = x.view(x.size(0), -1, 64)  # Flatten the output for LSTM
        x, _ = self.lstm(x)  # Output from LSTM
        x = self.fc1(x[:, -1, :])  # Take the last output of LSTM
        x = self.fc2(x)
        return x

__all__ = ["Base", "SVM", "SGDSVM", "Hybrid"]
