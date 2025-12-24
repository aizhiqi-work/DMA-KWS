from models.encoder import ConformerEncoder
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math


class PositionalEncoding(nn.Module):
    """位置编码模块"""
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class ModalityEmbedding(nn.Module):
    """模态类型嵌入，用于区分不同模态"""
    def __init__(self, d_model):
        super(ModalityEmbedding, self).__init__()
        self.text_emb = nn.Parameter(torch.randn(1, 1, d_model))
        self.audio_emb = nn.Parameter(torch.randn(1, 1, d_model))
        
    def forward(self, x, modality_type):
        batch_size, seq_len = x.size(0), x.size(1)
        if modality_type == 'text':
            return x + self.text_emb.expand(batch_size, seq_len, -1)
        elif modality_type == 'audio':
            return x + self.audio_emb.expand(batch_size, seq_len, -1)
        else:
            raise ValueError(f"不支持的模态类型: {modality_type}")


class GRUFCModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GRUFCModel, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        gru_out, _ = self.gru(x)
        gru_last_output = gru_out[:, -1, :]
        fc_out = self.fc(gru_last_output)
        return fc_out


class QbyT(nn.Module):
    def __init__(
        self,
        encoder_output_size=144,
        num_embeds=73, # 其中0,1,2均不会使用 <blank> <unk> <sos/eos>
        embed_dim=128,
        post_num_layers=2,
    ):
        super().__init__()
        self.audio_projection = nn.Linear(encoder_output_size, embed_dim)
        self.text_projection = nn.Embedding(num_embeddings=num_embeds, embedding_dim=embed_dim)
        self.pos_enc = PositionalEncoding(embed_dim)
        self.modality_enc = ModalityEmbedding(embed_dim)

        self.phone_matchor = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=4,
                dim_feedforward=embed_dim*4,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=post_num_layers
        )
        self.gru = nn.GRU(embed_dim, embed_dim, batch_first=True)
        self.fc = nn.Linear(embed_dim, 1)
        self.seq_fc = nn.Linear(embed_dim, 1)

    
    def forward(self, speech, text):
        # 文本处理
        text_emb = self.text_projection(text)
        text_emb = self.pos_enc(text_emb)
        text_emb = self.modality_enc(text_emb, 'text')

        # 音频处理
        audio_emb = self.audio_projection(speech)
        audio_emb = self.pos_enc(audio_emb)
        audio_emb = self.modality_enc(audio_emb, 'audio')

        combined_feat = torch.cat([text_emb, audio_emb], dim=1)
        combined_feat = self.phone_matchor(combined_feat)
        gru_out, _ = self.gru(combined_feat)
        gru_out = gru_out[:, -1, :]
        logits = self.fc(gru_out).squeeze(-1)

        text_logits = self.seq_fc(combined_feat[:, :text_emb.shape[1], :]).squeeze(-1)
        return logits, text_logits


if __name__ == "__main__":
    encoder = ConformerEncoder(
        input_size=80,
        output_size=144,
        attention_heads=4,
        linear_units=576,
        num_blocks=6,
        dropout_rate=0.1,
        positional_dropout_rate=0.1,
        attention_dropout_rate=0.0,
        use_cnn_module=True,
        input_layer="conv2d",
        pos_enc_layer_type="rel_pos",
        selfattention_layer_type="rel_selfattn",
        cnn_module_kernel=3,
    )

    model = QbyT()

    import torch
    feat = torch.randn(32, 204, 80)
    feat_lengths = torch.tensor([feat.shape[1]], device=feat.device)
    text = torch.randint(0, 71, (32, 12))
    print(text.shape)

    encoder_out, _ = encoder(feat, feat_lengths)
    logits, text_logits = model(encoder_out, text)


    print(logits.shape)
    print(text_logits.shape)