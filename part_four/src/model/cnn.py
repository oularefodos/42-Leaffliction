import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, in_channels=3, conv_hidden_channels=10, dense_hidden_size=256, output_size=10) -> None:
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=conv_hidden_channels,
                kernel_size=3,
            ),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=conv_hidden_channels,
                out_channels=conv_hidden_channels,
                kernel_size=3,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=conv_hidden_channels,
                out_channels=conv_hidden_channels,
                kernel_size=3,
            ),
            nn.ReLU(),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=conv_hidden_channels,
                out_channels=conv_hidden_channels,
                kernel_size=3,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.dense = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(dense_hidden_size),
            nn.ReLU(),
            nn.Linear(dense_hidden_size * 2, dense_hidden_size),
            nn.ReLU(),
            nn.Linear(dense_hidden_size, output_size)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.dense(x)
        return x
