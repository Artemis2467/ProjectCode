import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, d_model=5, max_len=150):
        super().__init__()

        pe = torch.zeros(max_len, d_model)

        self.position = torch.arange(start=0, end=max_len, step=1).float().unsqueeze(1)
        self.embedding_index = torch.arange(start=0, end=d_model, step=2).float()

        self.div_term = 1 / torch.tensor(10_000) ** (self.embedding_index / d_model)

        pe[:, 0::2] = torch.sin(self.position * self.div_term)
        pe[:, 1:d_model:2] = torch.cos(self.position * self.div_term[:d_model // 2])

        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:x.size(1), :].unsqueeze(0)

class Attention(nn.Module):
    def __init__(self, d_model=5):
        super().__init__()

        self.w_q = nn.Linear(in_features=d_model, out_features=32, bias=False)
        self.w_k = nn.Linear(in_features=d_model, out_features=32, bias=False)
        self.w_v = nn.Linear(in_features=d_model, out_features=32, bias=False)

        self.row_dim = 1
        self.col_dim = 2

    def forward(self, encodings_for_q, encodings_for_k, encodings_for_v):

        q = self.w_q(encodings_for_q)
        k = self.w_k(encodings_for_k)
        v = self.w_v(encodings_for_v)

        sims = torch.matmul(q, k.transpose(dim0=self.row_dim, dim1=self.col_dim))

        scaled_sims = sims / torch.tensor(k.size(self.col_dim) ** 0.5)

        attention_percents = F.softmax(scaled_sims, dim=self.col_dim)
        attention_scores = torch.matmul(attention_percents, v).transpose(1, 2)

        return attention_scores

class LogitModel(nn.Module):
    def __init__(self, out_features=1, d_model=5, max_len=150):
        super().__init__()

        self.pe = PositionalEncoding(d_model=d_model, max_len=max_len)
        self.self_attention = Attention(d_model=d_model)
        self.pooling_layer = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(p=0.2)
        self.fc_attention = nn.Linear(in_features=32, out_features=out_features)
        self.fc_final = nn.Linear(in_features=3, out_features=1)

    def forward(self, resp1_logits, resp2_logits, cosine_sim, perplexity):
        position1 = self.pe(resp1_logits)
        position2 = self.pe(resp2_logits)

        self_attention_values = self.self_attention(
            position1,
            position2,
            position2,
            )
        
        pooled = self.pooling_layer(self_attention_values)
        pooled = pooled.view(pooled.size(0), -1)

        dropped = self.dropout(pooled)
        attention_score = self.fc_attention(dropped)

        cosine_sim = cosine_sim.unsqueeze(1)
        perplexity = perplexity.unsqueeze(1)

        res = self.fc_final(torch.cat([attention_score, cosine_sim, perplexity], dim=1))

        return F.sigmoid(res)

class LinearModel(nn.Module):
    def __init__(self, d_model=32):
        super().__init__()

        self.fc1 = nn.Linear(in_features=2, out_features=d_model)
        self.batch_norm = nn.BatchNorm1d(num_features=d_model)
        self.fc2 = nn.Linear(in_features=d_model, out_features=1)
        
    
    def forward(self, cosine_sim, perplexity):

        combined = torch.cat([cosine_sim, perplexity], dim=1)
        linear_output1 = self.fc1(combined)
        normalized = self.batch_norm(linear_output1)
        linear_output2 = self.fc2(normalized)
        res = F.sigmoid(linear_output2)

        return res