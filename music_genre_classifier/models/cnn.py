from torch import nn

class CNNNetwork(nn.Module): 
    def __init__(self, num_classes=10):
        super().__init__()       
        
        # 1st conv layer
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, 
                      out_channels=32, 
                      kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3,
                         stride=2,
                         padding=1),
            nn.BatchNorm2d(32))

        # 2nd conv layer
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32,
                      out_channels=32,
                      kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3,
                         stride=2,
                         padding=1),
            nn.BatchNorm2d(32))
        
        # 3rd conv layer
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32,
                      out_channels=32,
                      kernel_size=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2,
                         padding=0),
            nn.BatchNorm2d(32))
        
        # Flatten output
        self.flatten = nn.Flatten()
        
        # Dense layer
        self.dense = nn.Sequential(
            nn.Linear(7200, 64),
            nn.ReLU())
        self.dropout = nn.Dropout(0.3)
        
        # Output layer
        self.output = nn.Linear(64, num_classes)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, input_data):
        x = self.conv1(input_data) 
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.dense(x)
        x = self.dropout(x)
        x = self.output(x)
        x = self.softmax(x)
        return x
    
if __name__ == '__main__':    
    cnn = CNNNetwork()