import torch
import torch.nn as nn
import torch.nn.functional as F

class Regression_MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, activation='sigmoid'):
        super(Regression_MLP, self).__init__()
        self.input_dim = input_dim
        if isinstance(hidden_dims, list) or isinstance(hidden_dims, tuple):
            self.hidden_dims = list(hidden_dims)
        elif isinstance(hidden_dims, int):
            self.hidden_dims = [hidden_dims]
        else:
            raise ValueError(f"hidden_dims must be a list or tuple of integers, or an integer, but got {type(hidden_dims)}")
        self.output_dim = output_dim

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden_dims[0]))
        self.layers.append(nn.Dropout(p=0.3))
        if len(hidden_dims) > 1:
            for i in range(1, len(hidden_dims)):
                self.layers.append(nn.Linear(hidden_dims[i-1], hidden_dims[i]))
                self.layers.append(nn.Dropout(p=0.3))
                self.layers.append(nn.LeakyReLU(0.3))
                self.layers.append(nn.LayerNorm(hidden_dims[i]))
        self.layers.append(nn.Linear(hidden_dims[-1], output_dim))
        self.n_layers = len(self.layers)
        self.activation = activation

    def forward(self, x):
        if x.dtype is not torch.float32:
            x = x.to(torch.float32)
        if len(x.shape) == 3:
            x = x.view(x.shape[0], -1)
        for i in range(self.n_layers):
            x = self.layers[i](x)
            if i == self.n_layers - 1:
                if self.activation == 'sigmoid':
                    x = F.sigmoid(x)
                elif self.activation == 'linear':
                    x = x
        return (x[:,0], x[:,1])

class Regression_MLP_Subscore(nn.Module):
    def __init__(self, input_dim, output_dim, moca_subscore_dim, mmse_subscore_dim, activation='sigmoid'):
        super(Regression_MLP_Subscore, self).__init__()
        self.input_dim = input_dim
        self.moca_subscore_dim = moca_subscore_dim
        self.mmse_subscore_dim = mmse_subscore_dim
        self.output_dim = output_dim
        self.activation = activation

        self.output_dim = output_dim

        hidden_dims = [4096, 1024, 256]
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden_dims[0]))
        self.layers.append(nn.LeakyReLU(0.3))
        self.layers.append(nn.Dropout(p=0.3))
        if len(hidden_dims) > 1:
            for i in range(1, len(hidden_dims)):
                self.layers.append(nn.Linear(hidden_dims[i-1], hidden_dims[i]))
                self.layers.append(nn.Dropout(p=0.3))
                self.layers.append(nn.LeakyReLU(0.3))
                self.layers.append(nn.LayerNorm(hidden_dims[i]))

        # output for moca and mmse scores
        self.output_layers = nn.ModuleList()
        self.output_layers.append(nn.Linear(256, 32))
        self.output_layers.append(nn.LayerNorm(32))
        self.output_layers.append(nn.Linear(32, output_dim))

        # output layers for subscores
        self.moca_subscore_layers = nn.ModuleList()
        self.moca_subscore_layers.append(nn.Linear(256, 32))
        self.moca_subscore_layers.append(nn.Dropout(p=0.3))
        self.moca_subscore_layers.append(nn.LeakyReLU(0.3))
        self.moca_subscore_layers.append(nn.LayerNorm(32))
        self.moca_subscore_layers.append(nn.Linear(32, moca_subscore_dim))

        # mmse subscore layers
        self.mmse_subscore_layers = nn.ModuleList()
        self.mmse_subscore_layers.append(nn.Linear(256, 32))
        self.mmse_subscore_layers.append(nn.Dropout(p=0.3))
        self.mmse_subscore_layers.append(nn.LeakyReLU(0.3))
        self.mmse_subscore_layers.append(nn.LayerNorm(32))
        self.mmse_subscore_layers.append(nn.Linear(32, mmse_subscore_dim))

        self.activation = activation

    def forward(self, x):
        if x.dtype is not torch.float32:
            x = x.to(torch.float32)
        if len(x.shape) == 3:
            x = x.view(x.shape[0], -1)

        for i in range(len(self.layers)):
            x = self.layers[i](x)

        output_embed = x
        moca_subscore_embed = x
        mmse_subscore_embed = x
        # output layers
        for i in range(len(self.output_layers)):
            output_embed = self.output_layers[i](output_embed)
            if i == len(self.output_layers) - 1:
                output_embed = F.sigmoid(output_embed)
            elif self.activation == 'linear':
                output_embed = output_embed

        for i in range(len(self.moca_subscore_layers)):
            moca_subscore_embed = self.moca_subscore_layers[i](moca_subscore_embed)
            if i == len(self.moca_subscore_layers) - 1:
                moca_subscore_embed = F.sigmoid(moca_subscore_embed)
            elif self.activation == 'linear':
                moca_subscore_embed = moca_subscore_embed

        for i in range(len(self.mmse_subscore_layers)):
            mmse_subscore_embed = self.mmse_subscore_layers[i](mmse_subscore_embed)
            if i == len(self.mmse_subscore_layers) - 1:
                mmse_subscore_embed = F.sigmoid(mmse_subscore_embed)
            elif self.activation == 'linear':
                mmse_subscore_embed = mmse_subscore_embed

        return (output_embed[:,0], output_embed[:,1], moca_subscore_embed, mmse_subscore_embed)

# class Regression_MLP_Subscore(nn.Module):
#     def __init__(self, input_dim, output_dim, moca_subscore_dim, mmse_subscore_dim, activation='sigmoid'):
#         super(Regression_MLP_Subscore, self).__init__()
#         self.input_dim = input_dim
#         self.moca_subscore_dim = moca_subscore_dim
#         self.mmse_subscore_dim = mmse_subscore_dim
#         self.output_dim = output_dim
#         self.activation = activation

#         self.output_dim = output_dim

#         hidden_dims = [1024, 512, 256]
#         self.layers = nn.ModuleList()
#         self.layers.append(nn.Linear(input_dim, hidden_dims[0]))
#         self.layers.append(nn.LeakyReLU(0.3))
#         self.layers.append(nn.Dropout(p=0.3))
#         if len(hidden_dims) > 1:
#             for i in range(1, len(hidden_dims)):
#                 self.layers.append(nn.Linear(hidden_dims[i-1], hidden_dims[i]))
#                 self.layers.append(nn.Dropout(p=0.3))
#                 self.layers.append(nn.LeakyReLU(0.3))
#                 self.layers.append(nn.LayerNorm(hidden_dims[i]))

#         # output for moca and mmse scores
#         self.output_layers = nn.ModuleList()
#         self.output_layers.append(nn.Linear(256, 64))
#         self.output_layers.append(nn.Dropout(p=0.3))
#         self.output_layers.append(nn.LeakyReLU(0.3))
#         self.output_layers.append(nn.LayerNorm(64))
#         self.output_layers.append(nn.Linear(64, 32))
#         self.output_layers.append(nn.Dropout(p=0.3))
#         self.output_layers.append(nn.LeakyReLU(0.3))
#         self.output_layers.append(nn.LayerNorm(32))
#         self.output_layers.append(nn.Linear(32, output_dim))

#         # output layers for subscores
#         self.moca_subscore_layers = nn.ModuleList()
#         self.moca_subscore_layers.append(nn.Linear(256, 64))
#         self.moca_subscore_layers.append(nn.Dropout(p=0.3))
#         self.moca_subscore_layers.append(nn.LeakyReLU(0.3))
#         self.moca_subscore_layers.append(nn.LayerNorm(64))
#         self.moca_subscore_layers.append(nn.Linear(64, 32))
#         self.moca_subscore_layers.append(nn.Dropout(p=0.3))
#         self.moca_subscore_layers.append(nn.LeakyReLU(0.3))
#         self.moca_subscore_layers.append(nn.LayerNorm(32))
#         self.moca_subscore_layers.append(nn.Linear(32, moca_subscore_dim))

#         # mmse subscore layers
#         self.mmse_subscore_layers = nn.ModuleList()
#         self.mmse_subscore_layers.append(nn.Linear(256, 64))
#         self.mmse_subscore_layers.append(nn.Dropout(p=0.3))
#         self.mmse_subscore_layers.append(nn.LeakyReLU(0.3))
#         self.mmse_subscore_layers.append(nn.LayerNorm(64))
#         self.mmse_subscore_layers.append(nn.Linear(64, 32))
#         self.mmse_subscore_layers.append(nn.Dropout(p=0.3))
#         self.mmse_subscore_layers.append(nn.LeakyReLU(0.3))
#         self.mmse_subscore_layers.append(nn.LayerNorm(32))
#         self.mmse_subscore_layers.append(nn.Linear(32, mmse_subscore_dim))

#         self.activation = activation

#     def forward(self, x):
#         if x.dtype is not torch.float32:
#             x = x.to(torch.float32)
#         if len(x.shape) == 3:
#             x = x.view(x.shape[0], -1)

#         for i in range(len(self.layers)):
#             x = self.layers[i](x)

#         output_embed = x
#         moca_subscore_embed = x
#         mmse_subscore_embed = x
#         # output layers
#         for i in range(len(self.output_layers)):
#             output_embed = self.output_layers[i](output_embed)
#             if i == len(self.output_layers) - 1:
#                 output_embed = F.sigmoid(output_embed)
#             elif self.activation == 'linear':
#                 output_embed = output_embed

#         for i in range(len(self.moca_subscore_layers)):
#             moca_subscore_embed = self.moca_subscore_layers[i](moca_subscore_embed)
#             if i == len(self.moca_subscore_layers) - 1:
#                 moca_subscore_embed = F.sigmoid(moca_subscore_embed)
#             elif self.activation == 'linear':
#                 moca_subscore_embed = moca_subscore_embed

#         for i in range(len(self.mmse_subscore_layers)):
#             mmse_subscore_embed = self.mmse_subscore_layers[i](mmse_subscore_embed)
#             if i == len(self.mmse_subscore_layers) - 1:
#                 mmse_subscore_embed = F.sigmoid(mmse_subscore_embed)
#             elif self.activation == 'linear':
#                 mmse_subscore_embed = mmse_subscore_embed

        # return (output_embed[:,0], output_embed[:,1], moca_subscore_embed, mmse_subscore_embed)


class Regression_MLP_Split_SubModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Regression_MLP_Split_SubModel, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, input_dim//2))
        self.layers.append(nn.Dropout(p=0.3))
        self.layers.append(nn.Linear(input_dim//2, output_dim))

    def forward(self, x):
        x = self.layers[0](x)
        x = self.layers[1](F.leaky_relu(x))
        x = self.layers[2](x)
        return F.leaky_relu(x)



class Regression_MLP_Split(nn.Module):
    def __init__(self, input_dim, feat_len, output_dim, activation='sigmoid'):
        super(Regression_MLP_Split, self).__init__()
        self.input_dim = input_dim
        self.feat_len = feat_len
        self.output_dim = output_dim

        self.n_task = input_dim//feat_len
        self.feat_len = feat_len
        
        n_task = input_dim//feat_len
        self.sub_models = nn.ModuleList()
        for i in range(n_task):
            self.sub_models.append(Regression_MLP_Split_SubModel(feat_len, feat_len//4))

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(feat_len//4, 256))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Dropout(p=0.3))
        self.layers.append(nn.Linear(256, 64))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Dropout(p=0.3))
        self.layers.append(nn.Linear(64, 16))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Dropout(p=0.3))
        self.layers.append(nn.Linear(16, output_dim))
        self.n_layers = len(self.layers)
        self.activation = activation

    def forward(self, x):
        if x.dtype is not torch.float32:
            x = x.to(torch.float32)
        x = x.view(-1, self.n_task, self.feat_len)
        sum_feat = torch.zeros(x.shape[0], self.feat_len//4).to(x.device)
        for i in range(self.n_task):
            sum_feat += self.sub_models[i](x[:,i,:])
        x = sum_feat.view(-1, self.feat_len//4)
        for i in range(self.n_layers):
            x = self.layers[i](x)
            if i == self.n_layers - 1:
                if self.activation == 'sigmoid':
                    x = F.sigmoid(x)
                elif self.activation == 'linear':
                    x = x
        return (x[:,0], x[:,1])

class Regression_CNN(nn.Module):
    def __init__(self, input_dim, output_dim, feat_len=1270, activation='sigmoid'):
        super(Regression_CNN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.n_task = input_dim//feat_len
        self.feat_len = feat_len
        self.layers = nn.ModuleList()
        self.layers.append(nn.Conv1d(in_channels=self.n_task, out_channels= self.n_task//2, kernel_size=16, stride=4))
        self.layers.append(nn.BatchNorm1d(self.n_task//2))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Dropout(p=0.3))
        self.layers.append(nn.Conv1d(in_channels=self.n_task//2, out_channels= self.n_task//4, kernel_size=16, stride=4))
        self.layers.append(nn.BatchNorm1d(self.n_task//4))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Dropout(p=0.3))
        self.layers.append(nn.Conv1d(in_channels=self.n_task//4, out_channels= self.n_task//8, kernel_size=16, stride=4))
        final_in_size = (((((self.feat_len-16)//4+1)-16)//4+1)-16)//4+1
        self.layers.append(nn.BatchNorm1d(self.n_task//8))
        
        self.final_layers = nn.Linear(final_in_size, output_dim)
        
        self.n_layers = len(self.layers)
        self.activation = activation

    def forward(self, x):
        if x.dtype is not torch.float32:
            x = x.to(torch.float32)
        x = x.view(-1, self.n_task, self.feat_len)
        for i in range(self.n_layers):
            x = self.layers[i](x)
            if i == self.n_layers - 1:
                x = self.final_layers(x).squeeze()
        if self.activation == 'sigmoid':
            x = F.sigmoid(x)
        elif self.activation == 'linear':
            x = x
        return (x[:,0], x[:,1])
    
class Regression_Transformer(nn.Module):
    def __init__(self, input_dim, output_dim, num_layers=2,activation='sigmoid'):
        super(Regression_Transformer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation = activation

        self.class_token = nn.Parameter(torch.randn(1, 1, input_dim))
        # self.pos_embed = nn.Parameter(torch.randn(1, 10 + 1, input_dim))

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=input_dim, nhead=1, dim_feedforward=input_dim*4, dropout=0.5),
            num_layers=num_layers
        )
        self.regressor = nn.Linear(input_dim, output_dim)
    def forward(self, x):
        ### x: [B, Task, Feature_dim]
        cls_tokens = self.class_token.repeat(x.shape[0], 1, 1)
        x = torch.cat((cls_tokens, x), dim=1)
        # x = x + self.pos_embed
        x = self.encoder(x)
        # x = x[:, 0]  # Take the cls token representation
        x = x.mean(dim=1)
        
        x = self.regressor(x)
        if self.activation == 'sigmoid':
            x = F.sigmoid(x)
        elif self.activation == 'linear':
            x = x
            
        return (x[:, 0], x[:, 1])

class Regression_Transformer_4head(nn.Module):
    def __init__(self, input_dim, output_dim, num_layers=2,activation='sigmoid'):
        super(Regression_Transformer_4head, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation = activation

        self.class_token = nn.Parameter(torch.randn(1, 1, input_dim))
        # self.pos_embed = nn.Parameter(torch.randn(1, 10 + 1, input_dim))

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=input_dim, nhead=4, dim_feedforward=input_dim*4, dropout=0.5, batch_first=True),
            num_layers=num_layers
        )
        self.regressor = nn.Linear(input_dim, output_dim)
    def forward(self, x):
        ### x: [B, Task, Feature_dim]
        cls_tokens = self.class_token.repeat(x.shape[0], 1, 1)
        x = torch.cat((cls_tokens, x), dim=1)
        # x = x + self.pos_embed
        x = self.encoder(x)
        # x = x[:, 0]  # Take the cls token representation
        x = x.mean(dim=1)
        
        x = self.regressor(x)
        if self.activation == 'sigmoid':
            x = F.sigmoid(x)
        elif self.activation == 'linear':
            x = x
            
        return (x[:, 0], x[:, 1])


class Regression_Transformer_TaskScore(nn.Module):
    def __init__(self, input_dim, output_dim,  n_layers=2, activation='sigmoid'):
        super(Regression_Transformer_TaskScore, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation = activation

        self.class_token = nn.Parameter(torch.randn(1, 1, input_dim))
        # self.pos_embed = nn.Parameter(torch.randn(1, 10 + 1, input_dim))

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=input_dim, nhead=1, dim_feedforward=input_dim*4, dropout=0.5, batch_first=True),
            num_layers=n_layers
        )
        self.regressor = nn.Linear(input_dim, output_dim)
        self.subscore_regressor = nn.Linear(input_dim, 13)

    def forward(self, x, src_key_padding_mask=None):
        ### x: [B, Task, Feature_dim]
        cls_tokens = self.class_token.repeat(x.shape[0], 1, 1)
        x = torch.cat((cls_tokens, x), dim=1)
        # x = x + self.pos_embed
        if src_key_padding_mask is not None:
            x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        else:
            x = self.encoder(x)
        # x = x[:, 0]  # Take the cls token representation
        x =x.mean(dim=1)
        
        latent_embedding = x

        x = self.regressor(x)
        subscore_x = self.subscore_regressor(latent_embedding)
        if self.activation == 'sigmoid':
            x = F.sigmoid(x)
            subscore_x = F.sigmoid(subscore_x)
        elif self.activation == 'linear':
            x = x
            subscore_x = subscore_x
            
        return (x[:, 0], x[:, 1], subscore_x[:, :7], subscore_x[:, 7:])


if __name__ == '__main__':
    # test the code
    # moca_pred = torch.randint(0, 30, (5,))
    # moca_gt = torch.randint(0, 30, (5,))
    # moca_pred_cls = torch.where(moca_pred >= 26, 0,
    #                              torch.where((moca_pred >= 18) & (moca_pred < 26), 1, 2))
    # moca_gt_cls = torch.where(moca_gt >= 26, 0,
    #                             torch.where((moca_gt >= 18) & (moca_gt < 26), 1, 2))
    # print(moca_pred)
    # print(moca_pred_cls)
    # print(moca_gt)
    # print(moca_gt_cls)

    model = Regression_MLP_Subscore(input_dim=200, output_dim=2, moca_subscore_dim=7, mmse_subscore_dim=6)
    x = torch.randn(5, 200)
    print(model(x))
    ret = model(x)
    print(ret[0].shape, max(ret[0]), min(ret[0]))
    print(ret[1].shape, max(ret[1]), min(ret[1]))
    print(ret[2].shape, ret[2], min(ret[2]))
    print(ret[3].shape, max(ret[3]), min(ret[3]))