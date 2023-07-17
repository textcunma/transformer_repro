from torch import nn, Tensor
from transformer_repro.model.multi_head_attention import MultiHeadAttention
from transformer_repro.model.pos_feed_forward import PositionwiseFeedForward


class EncoderLayer(nn.Module):
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
        super(EncoderLayer, self).__init__()
        # Multi-Head Attention
        self.self_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout)
        # Postionwise FFN(位置ごとのFFN)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner_hid, dropout)

    def forward(self, enc_input: Tensor, slf_attn_mask: Tensor = None) -> Tensor:
        """

        Args:
            enc_input (Tensor): shape [batch_size, max_length, d_model]
            slf_attn_mask (Tensor): shape [batch_size, len_q, len_k]

        Returns:
            enc_output (Tensor): shape [batch_size, max_length, d_model]

        """
        # 1uery,key,valueの値は同じもの(=enc_input)を使用
        enc_output = self.self_attn(
            q=enc_input, k=enc_input, v=enc_input, attn_mask=slf_attn_mask
        )
        enc_output = self.pos_ffn(enc_output)
        return enc_output  # type: ignore
