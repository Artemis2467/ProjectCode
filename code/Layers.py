import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self):
        super().__init__()
        token_num = 5
        max_len = 75

        pe = torch.zeros(max_len, token_num)

        self.position = torch.arange(start=0, end=max_len, step=1).float().unsqueeze(1)
        self.embedding_index = torch.arange(start=0, end=token_num, step=2).float()

        self.div_term = 1 / torch.tensor(10_000) ** (self.embedding_index / token_num)

        pe[:, 0::2] = torch.sin(self.position * self.div_term)
        pe[:, 1:token_num:2] = torch.cos(self.position * self.div_term[:token_num // 2])

        self.register_buffer("pe", pe)

    def forward(self, x):
        res = x + self.pe[:x.size(1), :].unsqueeze(0)
        return res

class Attention(nn.Module):
    def __init__(self, inner_feature):
        super().__init__()
        token_num = 5

        self.w_q = nn.Linear(in_features=token_num, out_features=inner_feature, bias=False)
        self.w_k = nn.Linear(in_features=token_num, out_features=inner_feature, bias=False)
        self.w_v = nn.Linear(in_features=token_num, out_features=inner_feature, bias=False)

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
    def __init__(self, config):
        super().__init__()

        self.con = config
        self.is_linear = False

        self.pe = PositionalEncoding()
        self.self_attention = Attention(inner_feature=config.d_model)

        if config.add_conv and config.conv_ch:
            self.conv = nn.Conv2d(1, config.conv_ch, kernel_size=(3, 3))
            self.pooling_layer = nn.AdaptiveMaxPool2d((1, 1))
            self.dropout = nn.Dropout(p=config.drop_out)
            self.fc_attention = nn.Linear(in_features=config.conv_ch, out_features=1)
        elif not config.add_conv:
            self.pooling_layer = nn.AdaptiveAvgPool1d(1)
            self.dropout = nn.Dropout(p=config.drop_out)
            self.fc_attention = nn.Linear(in_features=config.d_model, out_features=1)
        else:
            raise RuntimeError("add_conv not properly structured")
        
        self.fc_final = nn.Linear(in_features=3, out_features=1)


    def forward(self, resp1_logits, resp2_logits, cosine_sim, entropy):
        position1 = self.pe(resp1_logits)
        position2 = self.pe(resp2_logits)

        self_attention_values = self.self_attention(
            position1,
            position2,
            position2,
            )

        if self.con.add_conv:
            conv_output = self.conv(self_attention_values.unsqueeze(1))
            pooled = self.pooling_layer(conv_output).view(conv_output.size(0), conv_output.size(1))
        elif not self.con.add_conv:
            pooled = self.pooling_layer(self_attention_values)
            pooled = pooled.view(pooled.size(0), -1)
        else:
            raise RuntimeError("add_conv not properly structured")

        dropped = self.dropout(pooled)
        attention_score = self.fc_attention(dropped)

        cosine_sim = cosine_sim.unsqueeze(1)
        entropy = entropy.unsqueeze(1)

        res = self.fc_final(torch.cat([attention_score, cosine_sim, entropy], dim=1))

        return res

class LinearModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.is_linear = True

        self.fc1 = nn.Linear(in_features=2, out_features=config.d_model)
        self.batch_norm = nn.BatchNorm1d(num_features=config.d_model)
        self.fc2 = nn.Linear(in_features=config.d_model, out_features=1)
        
    
    def forward(self, cosine_sim, entropy):

        combined = torch.cat([cosine_sim.unsqueeze(1), entropy.unsqueeze(1)], dim=1)
        linear_output1 = self.fc1(combined)
        normalized = self.batch_norm(linear_output1)
        res = self.fc2(normalized)
        
        return res