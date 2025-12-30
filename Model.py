import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import lightning as L

class PositionalEncoding(nn.Module):
    def __init__(self, d_model=525, max_len=10):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
    
        position = torch.arange(start=0, end=max_len, step=1).float().unsqueeze(1)
        embedding_index = torch.arrange(start=0, end=d_model, step=2).float()

        div_term = 1 / torch.tensor(10_000) ** (embedding_index / d_model)

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe)

    def forward(self, word_embeddings):
        return word_embeddings + self.pe[:word_embeddings.size(0), :]


class Attention(nn.Module):
    def __init__(self, d_model=525):
        super().__init__()
    
        self.w_q = nn.Linear(in_features=d_model, out_features=d_model, bias=False)
        self.w_k = nn.Linear(in_features=d_model, out_features=d_model, bias=False)
        self.w_v = nn.Linear(in_features=d_model, out_features=d_model, bias=False)

        self.row_dim = 0
        self.col_dim = 1

    def forward(self, encodings_for_q, encodings_for_k, encodings_for_v, mask=None):

        q = self.w_q(encodings_for_q)
        k = self.w_k(encodings_for_k)
        v = self.w_v(encodings_for_v)

        sims = torch.matmul(q, k.transpose(dim0=self.row_dim, dim1=self.col_dim))

        scaled_sims = sims / torch.tensor(k.size(self.col_dim)**5)

        if mask:
            scaled_sims = scaled_sims.masked_fill(mask=mask, value=1e-9)
        
        attention_percents = F.softmax(scaled_sims, dim=self.col_dim)
        attention_scores = torch.matmul(attention_percents, v)

        return attention_scores
    
class Decoder(L.LightningModule):
    def __init__(self, num_tokens, d_model=525, max_len=10):
        super().__init__()

        self.we = nn.Embedding(num_embeddings=num_tokens, embedding_dim=d_model)
        self.pe = PositionalEncoding(d_model=d_model, max_len=max_len)
        self.self_attention = Attention(d_model=d_model)
        self.fc_layer = nn.Linear(in_features=d_model, out_features=num_tokens)

        self.loss = nn.CrossEntropyLoss()

    def forward(self, token_ids):
        word_embeddings = self.we(token_ids)
        position_encoded = self.pe(word_embeddings)

        mask = torch.zeros((token_ids.size(dim=0), token_ids.size(dim=0)))
        mask = mask == 0

        self_attention_values = self.self_attention(position_encoded,
                                                    position_encoded,
                                                    position_encoded,
                                                    mask=True,
                                                    )
        residual_connection_values = position_encoded + self_attention_values

        fc_layer_output = self.fc_layer(residual_connection_values)

        return fc_layer_output
    
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-3)
    
    def training_step(self, batch, batch_idx):
        input_tokens, labels = batch
        output = self.forward(input_tokens[0])
        loss = self.loss(output, labels[0])

        return loss

