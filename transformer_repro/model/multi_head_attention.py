import torch

from torch import nn, Tensor

from transformer_repro.model.scaled_dot_attention import ScaledDotProductAttention


class MultiHeadAttention(nn.Module):
    def __init__(
        self, n_head: int, d_model: int, d_k: int, d_v: int, dropout: float
    ) -> None:
        """

        Args:
             n_head (int): ヘッド数
             d_model (int): 入力次元数
             d_k (int): keyの次元数
             d_v (int): valueの次元数
             dropout (float): ドロップアウト率

        各ヘッドごとで異なる重みを用いて学習するために3つの行列を用意   ※ 内積を取るためクエリとキーの次元数は同じ
        w_qs: ヘッドごとのqueryベクトルを生成する行列 , size=(n_head, d_model, d_k)
        w_ks: ヘッドごとのkeyベクトルを生成する行列, size=(n_head, d_model, d_k)
        w_vs: ヘッドごとのvalueベクトルを生成する行列, size=(n_head, d_model, d_v)
        """
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.attention = ScaledDotProductAttention(d_model, dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout_value = dropout
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(d_v, d_model)
        nn.init.xavier_normal_(self.proj.weight)

        # クエリとキーは内積するため次元数は同じ
        self.linear_q = nn.Linear(d_model, d_k, bias=False)
        self.linear_k = nn.Linear(d_model, d_k, bias=False)
        self.linear_v = nn.Linear(d_model, d_v, bias=False)
        nn.init.xavier_normal_(self.linear_q.weight)
        nn.init.xavier_normal_(self.linear_k.weight)
        nn.init.xavier_normal_(self.linear_v.weight)

    def forward(
        self, q: Tensor, k: Tensor, v: Tensor, attn_mask: Tensor = None
    ) -> Tensor:
        """

        Args:
             q (Tensor): queryベクトル, shape [batch_size, len_q, d_model]
             k (Tensor): keyベクトル, shape [batch_size, len_k, d_model]
             v (Tensor): valueベクトル, shape [batch_size, len_v, d_model]
             attn_mask (Tensor): Attentionのマスク, shape [batch_size, len_q, len_k]

        Returns:
             outputs (Tensor): 各ヘッドの出力, shape [batch_size, len_q, d_model]
        """

        residual = q  # 残差接続用

        # 線形変換
        q = self.linear_q(q)  # [batch_size, len_q, d_model] -> [batch_size, len_q, d_k]
        k = self.linear_k(k)  # [batch_size, len_q, d_model] -> [batch_size, len_q, d_k]
        v = self.linear_v(v)  # [batch_size, len_q, d_model] -> [batch_size, len_q, d_v]

        # ヘッド分割
        q = self.split_head(
            q
        )  # [batch_size, len_q, d_k] -> [batch_size, head_num, len_q, d_k//head_num]
        k = self.split_head(
            k
        )  # [batch_size, len_q, d_k] -> [batch_size, head_num, len_q, d_k//head_num]
        v = self.split_head(
            v
        )  # [batch_size, len_q, d_v] -> [batch_size, head_num, len_q, d_v//head_num]

        # 各ヘッドごとに自己注意
        attn_mask = attn_mask.expand(  # type: ignore
            self.n_head, attn_mask.size(0), attn_mask.size(1), attn_mask.size(2)  # type: ignore
        )

        # PyTorch 2.0 から scaled dot attentionが利用可能になった。それを利用するのもOK
        outputs = self.attention(q, k, v, attn_mask=attn_mask)

        # ヘッドを連結
        outputs = self.concat_head(
            outputs
        )  # [batch_size, head_num, len, dim//head] -> [batch_size, len, dim_v]

        # 次元数を隠れ次元に戻す、残差接続
        outputs = self.proj(
            outputs
        )  # (batch_size, len_q, dim_v) -> (batch_size, len_q, d_model)

        outputs = self.dropout(outputs)
        outputs = self.layer_norm(outputs + residual)

        return outputs  # type: ignore

    def split_head(self, x: Tensor) -> Tensor:
        """ヘッド分割

        Args:
             x (Tensor): 入力テンソル size [batch_size, len, dim]

        Returns:
             x (Tensor): 出力テンソル size [batch_size, head_num, len, dim//head_num]
        """
        x = torch.tensor_split(x, self.n_head, dim=2)  # type: ignore   # [batch_size, len, dim] -> {Tuple: head_num} :::: [batch_size, len, dim//head_num] * head_num
        x = torch.stack(
            x, dim=1  # type: ignore
        )  # {Tuple: head_num} -> [batch_size, head_num, len, dim//head_num]
        return x

    @staticmethod
    def concat_head(x: Tensor) -> Tensor:
        """ヘッド連結

        Args:
             x (Tensor): 入力テンソル size [batch_size, head_num, len, dim//head]

        Returns:
             x (Tensor): 出力テンソル size [batch_size, len, dim_v]
        """
        x = torch.tensor_split(x, x.size()[1], dim=1)  # type: ignore
        x = torch.concat(x, dim=3).squeeze(dim=1)  # type: ignore
        return x
