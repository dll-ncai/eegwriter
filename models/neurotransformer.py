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

class Neurotransformer(nn.Module):
    def __init__(self):
        super(Neurotransformer, self).__init__()
        self.mffm_block1 = MFFMBlock(1)
        self.wave_block1 = WaveBlock(1, 25, 3, 5)
        self.mffm_block2 = MFFMBlock(32)
        self.wave_block2 = WaveBlock(32, 56, 3, 5)
        self.mffm_block3 = MFFMBlock(32)
        self.conv1 = nn.Conv1d(in_channels=25, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv3 = nn.Conv1d(in_channels=56, out_channels=32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(32)
        self.bn3 = nn.BatchNorm1d(32)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=32, nhead=8, dropout=0, batch_first=True),
            num_layers=3
        )
        # Transformer decoder for final pooling
        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=32, nhead=8, dropout=0, batch_first=True),
            num_layers=3
        )
        # Learnable classification token (1 token per sample)
        self.cls_token = nn.Parameter(torch.randn(1, 1, 32))
        self.fc = nn.Linear(32, 3)

        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.xavier_uniform_(self.conv3.weight)
        nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x):
        x1 = self.mffm_block1(x)
        x2 = self.wave_block1(x)
        x = x1 + x2

        # Apply spatial dropout
        x = F.dropout2d(x, 0.5, training=self.training)
        
        x = F.max_pool1d(x, kernel_size=2, stride=2)
        x = F.relu(self.bn1(self.conv1(x)))
        # x1 = self.mffm_block2(x)
        # x2 = self.wave_block2(x)
        # x = x1 + x2
        x = self.bn2(self.conv2(x))
        x = self.mffm_block3(x)
        x = F.max_pool1d(x, kernel_size=2, stride=2)
        x = self.bn3(self.conv3(x))
        
        # Permute to shape (batch_size, sequence_length, feature_dim)
        x = x.permute(0, 2, 1)
        x = self.encoder(x)
        
        # Use the transformer decoder with a learnable classification token
        batch_size = x.size(0)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # Shape: (B, 1, 32)
        # The decoder attends to the encoder output (memory) using the cls token as target.
        x = self.transformer_decoder(tgt=cls_tokens, memory=x)
        x = x.squeeze(1)  # Remove the sequence dimension
        
        x = self.fc(x)
        return x
