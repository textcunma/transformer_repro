import math

from torch import nn
from torch import Tensor

from transformer_repro.model.encoder_layer import EncoderLayer
from transformer_repro.utils.utils import get_attn_padding_mask
from transformer_repro.utils.utils import position_encoding_init


class Encoder(nn.Module):
    def __init__(
        self,
        n_src_vocab: int,
        max_len: int,
        n_layers: int,
        n_head: int,
        d_k: int,
        d_v: int,
        d_model: int,
        d_inner_hid: int,
        dropout: float,
        pad_idx: int,
    ) -> None:
        """

        Args:
            n_src_vocab (int): 入力言語の語彙数
            # embedding (): 埋め込み層
            max_len (int): 最大単語数
            n_layers (int): レイヤー数
            n_head (int): ヘッド数
            d_k (int): keyの次元数
            d_v (int): valueの次元数
            d_model(int): 入力次元数
            d_inner_hid (int): 隠れ層2層目の次元数
            n_head (int): ヘッド数
            dropout (float): ドロップアウト率
            pad_idx (int): paddingのインデックス
        """
        super(Encoder, self).__init__()

        # Embedding: ID -> Vector
        self.src_word_emb = nn.Embedding(n_src_vocab, d_model, padding_idx=pad_idx)

        # Positional Encoding
        n_pos = max_len + 1  # 23
        self.position_enc = nn.Embedding(n_pos, d_model, padding_idx=pad_idx)
        self.position_enc.weight.data = position_encoding_init(n_pos, d_model)

        self.pad_idx = pad_idx
        self.d_model = d_model

        # 指定レイヤー数から構成されるEncoderLayer
        self.layer_stack = nn.ModuleList(
            [
                EncoderLayer(d_model, d_inner_hid, n_head, d_k, d_v, dropout=dropout)
                for _ in range(n_layers)
            ]
        )

    def forward(self, src_seq: Tensor, src_pos: Tensor) -> Tensor:
        """

        Args:
            src_seq (Tensor): ID情報, shape [batch_size, max_length-1]
            src_pos (Tensor): 位置情報, shape [batch_size, max_length-1]

        Returns:
            enc_output(Tensor): 出力次元, shape [batch_size, max_length, d_model]
        """

        # Embedding
        enc_input = self.src_word_emb(src_seq) * math.sqrt(self.d_model)

        # Positional Embedding
        enc_input += self.position_enc(src_pos)

        enc_output = enc_input

        # key(=enc_input)のPADに対応する部分のみ1のマスクを作成
        enc_slf_attn_mask = get_attn_padding_mask(src_seq, src_seq, self.pad_idx)

        for enc_layer in self.layer_stack:
            enc_output = enc_layer(enc_output, slf_attn_mask=enc_slf_attn_mask)

        return enc_output  # type: ignore
