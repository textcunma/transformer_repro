import numpy as np

from torch import nn, Tensor


class ScaledDotProductAttention(nn.Module):
    """
    PyTorch 2.0からscaled dot attentionがライブラリ関数として利用可能になったが再現実装のため、今回は使用しない
    """

    def __init__(self, d_model: int, dropout: float) -> None:
        """

        Args:
             d_model (int): 隠れ層の次元数
             dropout (float): ドロップアウト率
        """
        super(ScaledDotProductAttention, self).__init__()
        self.temper = np.power(d_model, 0.5)  # スケーリング因子
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q: Tensor, k: Tensor, v: Tensor, attn_mask: Tensor) -> Tensor:
        """

        Args:
             q (Tensor): queryベクトル, shape [batch_size, head_num, len_q, d_k//head_num]
             k (Tensor): keyベクトル, shape [batch_size, head_num, len_q, d_k//head_num]
             v (Tensor): valueベクトル, shape [batch_size, head_num, len_q, d_v//head_num]
             attn_mask (Tensor): Attentionに適用するマスク, shape [n_head*batch_size, len_q, len_k]

        Returns:
             output (Tensor): 出力ベクトル, shape [n_head*batch_size, len_q, d_v]
        """
        # queryとkeyの内積でAttentionの重みを計算 -> スケーリング
        attn = (
            q @ k.transpose(3, 2) / self.temper
        )  # output: [batch_size, head_num, len_q, len_k]

        # Attentionをかけたくない部分を負の無限大に飛ばしてSoftmaxの値を0にする
        attn.data.masked_fill_(attn_mask.transpose(1, 0), -float("inf"))

        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = attn @ v  # [batch_size, head_num, len_q, len_k]
        return output  # [batch_size, head_num, len, d_v//head_num]
