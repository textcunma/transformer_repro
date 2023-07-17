from torch import nn, Tensor


class PositionwiseFeedForward(nn.Module):
    """位置ごとのFeed Forward"""

    def __init__(self, d_hid: int, d_inner_hid: int, dropout: float) -> None:
        """

        Args:
            d_hid (int): 隠れ層1層目の次元数
            d_inner_hid (int): 隠れ層2層目の次元数
            dropout (float): ドロップアウト率
        """
        super(PositionwiseFeedForward, self).__init__()

        # 畳み込み層のカーネルサイズを1にすると楽に実装可能
        self.w_1 = nn.Conv1d(d_hid, d_inner_hid, 1)
        self.w_2 = nn.Conv1d(d_inner_hid, d_hid, 1)
        self.layer_norm = nn.LayerNorm(d_hid)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, input: Tensor) -> Tensor:
        """

        Args:
            input (Tensor): 入力テンソル, shape [batch_size, max_length, d_hid]

        Returns:
            output (Tensor): 出力テンソル, shape [batch_size, max_length, d_hid]
        """
        residual = input
        output = self.relu(self.w_1(input.transpose(1, 2)))
        output = self.w_2(output).transpose(2, 1)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)
        return output  # type: ignore
