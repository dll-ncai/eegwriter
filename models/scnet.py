import torch
import torch.nn as nn
import torch.nn.functional as F

class MFFMBlock(nn.Module):
    def __init__(self, in_channels):
        super(MFFMBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=8, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(8)
        self.conv2 = nn.Conv1d(in_channels=in_channels + 8, out_channels=16, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(16)

        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)


    def forward(self, input):
        x1 = F.relu(self.bn1(self.conv1(input)))
        x1 = torch.cat((x1, input), dim=1)
        x2 = F.relu(self.bn2(self.conv2(x1)))
        return torch.cat((x2, x1), dim=1)

class SILM(nn.Module):
    def __init__(self):
        super(SILM, self).__init__()

    def forward(self, x):
        gap = torch.mean(x, dim=1, keepdim=True)
        gsp = torch.std(x, dim=1, keepdim=True)
        gmp, _ = torch.max(x, dim=1, keepdim=True)
        gap = F.dropout(gap, 0.05, training=self.training)
        gmp = F.dropout(gmp, 0.05, training=self.training)
        gsp = F.dropout(gsp, 0.05, training=self.training)
        x = torch.cat((x, gap, gsp, gmp), dim=1)
        return x


class SCNet(nn.Module):
    def __init__(self, input_shape):
        super(SCNet, self).__init__()
        self.silm = SILM()
        self.mffm_block1 = MFFMBlock(50)
        self.mffm_block2 = MFFMBlock(50)
        self.mffm_block3 = MFFMBlock(32)
        self.mffm_block4 = MFFMBlock(32)
        self.mffm_block5 = MFFMBlock(32)
        self.conv1 = nn.Conv1d(in_channels=74, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(50)
        self.conv2 = nn.Conv1d(in_channels=56, out_channels=32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(32)
        self.conv3 = nn.Conv1d(in_channels=56, out_channels=32, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(32)
        self.fc = nn.Linear(32, 2)

        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.xavier_uniform_(self.conv3.weight)
        nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x):
        x = self.silm(x)
        x1 = F.avg_pool1d(x, kernel_size=2, stride=2)
        x2 = F.max_pool1d(x, kernel_size=2, stride=2)
        x = torch.cat((x1, x2), dim=1)
        x = self.bn1(x)
        x1 = self.mffm_block1(x)
        x2 = self.mffm_block2(x)
        x = x1 + x2

        #Apply spatial dropout
        # x = x.permute(0, 2, 1)
        x = F.dropout2d(x, 0.5, training=self.training)
        # x = x.permute(0, 2, 1)

        x = F.max_pool1d(x, kernel_size=2, stride=2)
        x = F.relu(self.bn2(self.conv1(x)))
        x1 = self.mffm_block3(x)
        x2 = self.mffm_block4(x)
        x = x1 + x2
        x = self.bn3(self.conv2(x))
        x = self.mffm_block5(x)
        x = F.max_pool1d(x, kernel_size=2, stride=2)
        x = self.conv3(x)
        x = torch.mean(x, dim=2)
        x = self.fc(x)
        # return F.softmax(x, dim=-1)
        return x

