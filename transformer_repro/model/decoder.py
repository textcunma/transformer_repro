import torch

from torch import nn, Tensor
from transformer_repro.model.decoder_layer import DecoderLayer
from transformer_repro.utils.utils import get_attn_padding_mask
from transformer_repro.utils.utils import position_encoding_init
from transformer_repro.utils.utils import get_attn_subsequent_mask


class Decoder(nn.Module):
    def __init__(
        self,
        n_tgt_vocab: int,
        max_len: int,
        n_layers: int,
        n_head: int,
        d_k: int,
        d_v: int,
        d_model: int,
        d_inner_hid: int,
        dropout: float,
        pad_idx: int,
        device: torch.device,
    ) -> None:
        """
        Args:
                n_tgt_vocab (int): 出力言語の語彙数
                max_len (int): 最大単語数
                n_layers (int): レイヤー数
                n_head (int): ヘッド数
                d_k (int): keyの次元数
                d_v (int): valueの次元数
                d_model (int): 入力次元数
                d_inner_hid (int): Position Wise Feed Forward Networkの隠れ層2層目の次元数
                n_head (int): ヘッド数
                dropout (float): ドロップアウト率
                pad_idx (int): paddingのインデックス
                device (torch.device): GPUのデバイス
        """
        super(Decoder, self).__init__()

        # Embedding: ID -> Vector
        self.tgt_word_emb = nn.Embedding(n_tgt_vocab, d_model, padding_idx=pad_idx)

        # Positional Encoding
        n_pos = max_len + 1  # 23
        self.position_enc = nn.Embedding(n_pos, d_model, padding_idx=pad_idx)
        self.position_enc.weight.data = position_encoding_init(n_pos, d_model)

        self.max_len = max_len
        self.d_model = d_model
        self.pad_idx = pad_idx
        self.device = device

        self.layer_stack = nn.ModuleList(
            [
                DecoderLayer(d_model, d_inner_hid, n_head, d_k, d_v, dropout=dropout)
                for _ in range(n_layers)
            ]
        )

    def forward(
        self, tgt_seq: Tensor, tgt_pos: Tensor, src_seq: Tensor, enc_output: Tensor
    ) -> Tensor:
        """

        Args:
                tgt_seq (Tensor): 出力系列, shape [batch_size, max_length]
                tgt_pos (Tensor): 出力位置系列, shape [batch_size, max_length]
                src_seq (Tensor): 入力系列 マスクを作るために必要, shape [batch_size, n_src_vocab]
                enc_output (Tensor): Encoderの出力, shape [batch_size, max_length, d_model]

        Returns:
                dec_output (Tensor): shape [batch_size, max_length, d_model]
        """
        # Embedding
        dec_input = self.tgt_word_emb(tgt_seq)  # 生成途中の文をEmbeddingする

        # Positional Encoding
        dec_input += self.position_enc(tgt_pos)

        # Self-Attention用のマスクを作成
        # key(=dec_input)のPADに対応する部分が1のマスクと、queryから見たkeyの未来の情報に対応する部分が1のマスクのORをとる
        dec_slf_attn_pad_mask = get_attn_padding_mask(
            tgt_seq, tgt_seq, self.pad_idx
        )  # (N, max_length, max_length)
        dec_slf_attn_sub_mask = get_attn_subsequent_mask(
            tgt_seq, self.device
        )  # (N, max_length, max_length)
        dec_slf_attn_mask = torch.gt(
            dec_slf_attn_pad_mask + dec_slf_attn_sub_mask, 0
        )  # ORをとる

        # key(=dec_input)のPADに対応する部分のみ1のマスクを作成
        dec_enc_attn_pad_mask = get_attn_padding_mask(
            tgt_seq, src_seq, self.pad_idx
        )  # (N, max_length, max_length)

        dec_output = dec_input

        # n_layers個のDecoderLayerに入力を通す
        for dec_layer in self.layer_stack:
            dec_output = dec_layer(
                dec_output,
                enc_output,
                slf_attn_mask=dec_slf_attn_mask,
                dec_enc_attn_mask=dec_enc_attn_pad_mask,
            )

        return dec_output  # type: ignore
