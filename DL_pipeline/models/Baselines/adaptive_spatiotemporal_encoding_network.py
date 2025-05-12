import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        return output + self.bias if self.bias is not None else output

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        return self.gc2(x, adj)

class CombinedGCNCNN(nn.Module):
    def __init__(self, signal_len=125*3*60, n_channels=16, output_dim=2, CLS_or_Reg='Reg'):
        super(CombinedGCNCNN, self).__init__()
        self.signal_len = signal_len
        self.n_channels = n_channels
        self.relu = nn.ReLU()

        self.gcn = GCN(nfeat=signal_len, nhid=100, nclass=signal_len, dropout=0.3)

        self.conv1 = nn.Conv1d(n_channels, 32, kernel_size=5, stride=2, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2)
        self.bn3 = nn.BatchNorm1d(128)

        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=128, nhead=4, dim_feedforward=256, dropout=0.2),
            num_layers=2
        )

        with torch.no_grad():
            dummy_out = self._forward_cnn_only(torch.zeros(1, n_channels, signal_len))
            self.flatten_dim = dummy_out.reshape(1, -1).shape[1]

        self.fc1 = nn.Linear(self.flatten_dim, 256)
        self.bn_fc1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, output_dim)
        self.cls_or_reg = CLS_or_Reg
        if CLS_or_Reg.lower() == 'reg':
            self.last_act = nn.Sigmoid()
        else:
            self.last_act = nn.Identity()

    def _forward_cnn_only(self, x):
        x = self.bn1(self.relu(self.conv1(x)))
        x = self.bn2(self.relu(self.conv2(x)))
        x = self.bn3(self.relu(self.conv3(x)))
        x = x.permute(0, 2, 1)
        x = self.transformer_encoder(x)
        return x.permute(0, 2, 1)

    def forward(self, x):
        if isinstance(x, list):
            exg, dtf = x
            dtf = dtf.mean(-1)
            x = exg
        B, C, L = x.shape
        gcn_out = self.gcn(x, dtf)
        x = x + gcn_out

        x = self.bn1(self.relu(self.conv1(x)))
        x = self.bn2(self.relu(self.conv2(x)))
        x = self.bn3(self.relu(self.conv3(x)))

        x = x.permute(0, 2, 1)
        x = self.transformer_encoder(x)
        x = x.permute(0, 2, 1)

        x = x.reshape(B, -1)
        x = self.bn_fc1(self.relu(self.fc1(x)))
        x = self.last_act(self.fc2(x))
        if self.cls_or_reg.lower() == 'reg':
            return x[:, 0], x[:, 1]
        else:
            return x, 0

# # ------------------ Test ------------------

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ExG = torch.randn((16, 16, 125*3*60)).to(device)
# DTF = torch.randn((16, 16, 16)).to(device)

# model = CombinedGCNCNN().to(device)
# output = model(ExG, DTF)
# print("Output shape:", output.shape)  # Expected: [16, 2]
