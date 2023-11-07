import torch
import torch.nn as nn
import math

class simpleTransformer(nn.Module):
    def __init__(self, max_len, arch, shrink) -> None:
        super().__init__()
        d_model, nhead, num_layers = tuple(map(lambda x: int(x), arch.split("-")))
        self.emb = nn.Embedding(num_embeddings=3, embedding_dim=d_model)

        self.make_pos_emb(d_model, max_len+1)   # consider cls

        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model*4, dropout=0.0, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer=layer, num_layers=num_layers)

        self.head = nn.Linear(d_model, 1)

        for p in self.parameters():
            p.data = p.data * shrink

    def make_pos_emb(self, d_model, max_len):

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, inputs):
        # inputs: batch_size, seq_len
        cls_token = torch.full((inputs.size(0), 1), fill_value=2, dtype=inputs.dtype, device=inputs.device)
        x = torch.hstack([inputs, cls_token])

        x = self.emb(x)
        x = x + self.pe
        x = self.encoder(x) # batch_size, seq_len+1, d_model

        x = x[:, -1, :]
        # x = torch.sigmoid(self.head(x).squeeze(-1))
        x = self.head(x).squeeze(-1)

        return x
    
class simpleLSTM(nn.Module):
    def __init__(self, max_len, d_model=64, num_layers=3) -> None:
        super().__init__()

        self.emb = nn.Embedding(num_embeddings=3, embedding_dim=d_model)

        self.lstm = nn.LSTM(d_model, d_model, num_layers, batch_first=True)

        self.head = nn.Linear(d_model, 1)

    def forward(self, inputs):
        # inputs: batch_size, seq_len
        cls_token = torch.full((inputs.size(0), 1), fill_value=2, dtype=inputs.dtype, device=inputs.device)
        x = torch.hstack([inputs, cls_token])

        x = self.emb(x)
        x, (h, c) = self.lstm(x) # batch_size, seq_len+1, d_model

        x = x[:, -1, :]
        x = torch.sigmoid(self.head(x).squeeze(-1))

        return x
    
class DNN(nn.Module):
    def __init__(self, max_len, arch, shrink) -> None:
        super().__init__()
        hidden_d, num_layers = tuple(map(lambda x: int(x), arch.split("-")))

        assert num_layers >= 2

        layers = [nn.Linear(max_len, hidden_d)]
        for i in range(num_layers-2):
            layers.append(nn.ReLU())
            layers.append(nn.Linear(hidden_d, hidden_d))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_d, 1))

        self.net = nn.Sequential(*layers)
        # for p in self.parameters():
        #     p.data = torch.zeros_like(p.data)

    def forward(self, inputs):
        # inputs: batch_size, seq_len
        return self.net(inputs.float()).squeeze(-1)

modelClass = {"transformer": simpleTransformer, "lstm": simpleLSTM, "dnn": DNN}
optClass = {'sgd': torch.optim.SGD, 'adam': torch.optim.Adam, 'adamw': torch.optim.AdamW}