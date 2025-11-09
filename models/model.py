"""
CNN model to classify cats and dogs
"""

from torch import nn
from torch.nn import functional as F
from huggingface_hub import PyTorchModelHubMixin


class CnnModel(
    nn.Module,
    PyTorchModelHubMixin,
    repo_url="https://github.com/sachinkum0009/cat-vs-dog.git",
):
    """
    Basic CNN Model for classification task
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=16, kernel_size=(5, 5), stride=2, padding=1
        )
        self.conv2 = nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=(5, 5), stride=2, padding=1
        )
        self.conv3 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1
        )

        self.fc1 = nn.Linear(in_features=64 * 6 * 6, out_features=500)
        self.fc2 = nn.Linear(in_features=500, out_features=50)
        self.fc3 = nn.Linear(in_features=50, out_features=2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)

        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

