import torch.nn as nn

from torch import Tensor
from transformer_repro.model.multi_head_attention import MultiHeadAttention
from transformer_repro.model.pos_feed_forward import PositionwiseFeedForward


class DecoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_inner_hid: int,
        n_head: int,
        d_k: int,
        d_v: int,
        dropout: float,
    ) -> None:
        """

        Args:
             d_model (int): 入力次元数
             d_inner_hid (int): 隠れ層2層目の次元数
             n_head (int): ヘッド数
             d_k (int): keyの次元数
             d_v (int): valueの次元数
             dropout (float): ドロップアウト率
        """
        super(DecoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        # Encoder-Decoder間のSource-Target Attention
        self.enc_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)

        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner_hid, dropout=dropout)

    def forward(
        self,
        dec_input: Tensor,
        enc_output: Tensor,
        slf_attn_mask=None,
        dec_enc_attn_mask=None,
    ) -> Tensor:
        """

        Args:
             dec_input (Tensor): Decoderの入力, size=(batch_size, max_length, d_model)
             enc_output (Tensor): Encoderの出力, size=(batch_size, max_length, d_model)
             slf_attn_mask (Tensor): Self Attentionの行列にかけるマスク, size=(batch_size, len_q, len_k)
             dec_enc_attn_mask (Tensor): Soutce-Target Attentionの行列にかけるマスク, size=(batch_size, len_q, len_k)

        Returns:
             dec_output (Tensor): Decoderの出力, size=(batch_size, max_length, d_model)
        """

        # Self-Attentionのquery, key, valueにはすべてDecoderの入力（dec_input）が入る
        # 日本語同士の関係を加味する
        dec_output = self.slf_attn(
            dec_input, dec_input, dec_input, attn_mask=slf_attn_mask
        )

        # Source-Target-AttentionのqueryにはDecoderの出力(dec_output), key, valueにはEncoderの出力（enc_output）が入る
        # 英語の関係を加味するsorce target attention
        # クエリとして「dec_output」、キー、バリューとして「enc_output」を使う
        dec_output = self.enc_attn(
            dec_output, enc_output, enc_output, attn_mask=dec_enc_attn_mask
        )
        dec_output = self.pos_ffn(dec_output)
        return dec_output  # type: ignore
