import torch
import torch.nn as nn
import torch.nn.functional as F

class WaveLayer(nn.Module):
    def __init__(self, in_channels, kernel_size, dilation):
        super(WaveLayer, self).__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, in_channels, kernel_size, padding=self.padding, dilation=dilation)
        self.tanh = nn.Tanh()
        self.sig = nn.Sigmoid()
        self.filter = nn.Conv1d(in_channels, in_channels, 1)
        self.gate = nn.Conv1d(in_channels, in_channels, 1)
        self.conv2 = nn.Conv1d(in_channels, in_channels, 1)

        # Initialize weights
        torch.nn.init.xavier_uniform_(self.conv.weight, gain=1.0)
        torch.nn.init.xavier_uniform_(self.filter.weight, gain=1.0)
        torch.nn.init.xavier_uniform_(self.gate.weight, gain=1.0)
        torch.nn.init.xavier_uniform_(self.conv2.weight, gain=1.0)
       # self.skip = nn.Conv1d(out_channels, in_channels, 1)
       # self.residual = nn.Conv1d(out_channels, in_channels, 1)
        
    def forward(self, x):
        # x_padded = torch.nn.functional.pad(x, (self.padding, 0))
        output = self.conv(x)
        filter = self.filter(output)
        gate = self.gate(output)
        tanh = self.tanh(filter)
        sig = self.sig(gate)
        z = tanh*sig
        z = z[:,:,:-self.padding]
        z = self.conv2(z)
        x = x + z
        return x

class WaveBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation_rates):
        super(WaveBlock, self).__init__()
        self.layers = nn.ModuleList()
        dilations = [2**i for i in range(dilation_rates)]
        self.conv1d = nn.Conv1d(in_channels, out_channels, 1)
        for dilation in dilations:
            self.layers.append(WaveLayer(out_channels, kernel_size, dilation))
        torch.nn.init.xavier_uniform_(self.conv1d.weight, gain=1.0)

    def forward(self, x):
        x = self.conv1d(x)
        for layer in self.layers:
            x = layer(x)
        return x

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

class NeuroGate(nn.Module):
    def __init__(self):
        super(NeuroGate, self).__init__()
        self.mffm_block1 = MFFMBlock(44)
        self.wave_block1 = WaveBlock(44, 68, 3, 8)
        self.mffm_block2 = MFFMBlock(20)
        self.wave_block2 = WaveBlock(20, 20 + 24, 3, 8)
        self.mffm_block3 = MFFMBlock(20)
        self.conv1 = nn.Conv1d(in_channels=68, out_channels=20, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(44)
        self.conv2 = nn.Conv1d(in_channels=20 + 24, out_channels=20, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(20)
        self.conv3 = nn.Conv1d(in_channels=20 + 24, out_channels=20, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(20)
        self.bn4 = nn.BatchNorm1d(20)
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(20, 4, dropout=0.5, batch_first=True), 2)
        self.fc = nn.Linear(20, 2)

        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.xavier_uniform_(self.conv3.weight)
        nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x):
        # gap = torch.mean(x, dim=1, keepdim=True)
        # gsp = torch.std(x, dim=1, keepdim=True)
        # gmp, _ = torch.max(x, dim=1, keepdim=True)
        # gmnp, _ = torch.min(x, dim=1, keepdim=True)
        # axis = 1  # compute kurtosis over the “channels” dimension

        # # 1) mean per example & channel
        # mu = x.mean(dim=axis, keepdim=True)

        # # 2) second moment (variance)
        # m2 = ((x - mu) ** 2).mean(dim=axis, keepdim=True)

        # # 3) fourth moment
        # m4 = ((x - mu) ** 4).mean(dim=axis, keepdim=True)

        # # 4) kurtosis: m4 / (m2**2) minus 3
        # gkp = m4.div(m2.pow(2)).sub(3.0)  # shape [batch, 1, ...]

        # gap = F.dropout(gap, 0.05, training=self.training)
        # gmp = F.dropout(gmp, 0.05, training=self.training)
        # gsp = F.dropout(gsp, 0.05, training=self.training)
        # gmnp = F.dropout(gmnp, 0.05, training=self.training)
        # gkp = F.dropout(gkp, 0.05, training=self.training)

        # x = torch.cat((x, gap, gsp, gmp, gmnp,gkp), dim=1)

        x1 = F.avg_pool1d(x, kernel_size=5, stride=5)
        x2 = F.max_pool1d(x, kernel_size=5, stride=5)
        x = torch.cat((x1, x2), dim=1)
        x = self.bn1(x)
        x1 = self.mffm_block1(x)
        x2 = self.wave_block1(x)
        x = x1 + x2

        #Apply spatial dropout
        # x = x.permute(0, 2, 1)
        x = F.dropout2d(x, 0.5, training=self.training)
        # x = x.permute(0, 2, 1)
        
        x = F.max_pool1d(x, kernel_size=5, stride=5)
        x = F.relu(self.bn2(self.conv1(x)))
        x1 = self.mffm_block2(x)
        x2 = self.wave_block2(x)
        x = x1 + x2
        x = self.bn3(self.conv2(x))
        x = self.mffm_block3(x)
        x = F.max_pool1d(x, kernel_size=5, stride=5)
        x = self.bn4(self.conv3(x))

        x = x.permute(0, 2, 1)
        x = self.encoder(x)
        x = x.permute(0, 2, 1)

        x = torch.mean(x, dim=2)
        x = self.fc(x)
        # return F.softmax(x, dim=-1)
        return x
