import torch.nn as nn

class PetLogisticRegression(nn.Module):
    def __init__(self):
        super(PetLogisticRegression, self).__init__()
        # Input: 3 channels x 224 height x 224 width
        self.fc = nn.Linear(3 * 224 * 224, 1)

    def forward(self, x):
        # Flatten the image pixels
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
