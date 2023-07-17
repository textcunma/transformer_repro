import tqdm
import torch
import argparse

from typing import Tuple
from torch import nn, Tensor

# 自作クラス
from transformer_repro.model.encoder import Encoder
from transformer_repro.model.decoder import Decoder

class Transformer(nn.Module):
    def __init__(
        self,
        args: argparse.Namespace,
        vocab_size: Tuple[int, int],
    ) -> None:
        """

        Args:
            args (argparse.Namespace): コマンドライン引数
            vocab_size (Tuple[int, int]): [英語語彙数, 日本語語彙数]
        """
        super(Transformer, self).__init__()

        self.encoder = Encoder(
            n_src_vocab=vocab_size[0],  # 入力言語の語彙数
            max_len=args.max_len,  # 最大系列長
            n_layers=args.n_layer,  # レイヤー数
            n_head=args.n_head,  # ヘッド数
            d_k=args.d_key,  # keyの出力ベクトルの次元数
            d_v=args.d_value,  # valueの出力ベクトルの次元数
            d_model=args.d_model,  # 隠れ層の次元数
            d_inner_hid=args.d_inner_hid,  # Feed-Forwardの隠れ層2層目の次元数
            dropout=args.dropout,  # ドロップアウト率
            pad_idx=args.pad_idx,  # パディング用のID
        )

        self.decoder = Decoder(
            n_tgt_vocab=vocab_size[1],  # 出力言語の語彙数
            max_len=args.max_len,
            n_layers=args.n_layer,
            n_head=args.n_head,
            d_k=args.d_key,  # keyの出力ベクトルの次元数
            d_v=args.d_value,  # valueの出力ベクトルの次元数
            d_model=args.d_model,
            d_inner_hid=args.d_inner_hid,
            dropout=args.dropout,
            pad_idx=args.pad_idx,
            device=args.device,  # GPUを使うかどうか
        )

        self.tgt_word_proj = nn.Linear(args.d_model, vocab_size[1], bias=False)
        nn.init.xavier_normal_(self.tgt_word_proj.weight)
        self.device = args.device
        self.batch_size = args.batch_size
        self.bos_idx = args.bos_idx
        self.eos_idx = args.eos_idx
        self.max_len = args.max_len

        if args.proj_share_weight:
            # 出力言語の単語のEmbeddingと出力の写像で重みを共有
            self.tgt_word_proj.weight = self.decoder.tgt_word_emb.weight

    def forward(self, src: Tuple[Tensor, Tensor], tgt: Tuple[Tensor, Tensor]) -> Tensor:
        """

        Args:
            src (Tuple[Tensor, Tensor]): [英語シーケンス情報, 英語位置情報]
            tgt (Tuple[Tensor, Tensor]): [日本語シーケンス情報, 日本語位置情報]

        Returns:
            seq_logit (torch.Tensor): 語彙出現確率, size(batch, max_len - 1, vocab_size)
        """
        src_seq, src_pos = src
        tgt_seq, tgt_pos = tgt

        src_seq = src_seq[:, 1:]  # <bos>を取り払う
        src_pos = src_pos[:, 1:]
        tgt_seq = tgt_seq[:, :-1]  # <eos>を取り払う
        tgt_pos = tgt_pos[:, :-1]

        enc_output = self.encoder(src_seq, src_pos)
        dec_output = self.decoder(tgt_seq, tgt_pos, src_seq, enc_output)

        # 確率分布の対数を返す
        seq_logit = self.tgt_word_proj(
            dec_output
        )  # 損失の計算にnn.CrossEntropyLoss()を使用する為、Softmax層を挿入しない

        return seq_logit  # type: ignore


class TransformerPredict(nn.Module):
    def __init__(self, args: argparse.Namespace, vocab_size: Tuple[int, int]) -> None:
        """
        Args:
            args (argparse.Namespace): コマンドライン引数
            vocab_size (Tuple[int, int]): [英語語彙数, 日本語語彙数]
        """
        super(TransformerPredict, self).__init__()

        self.encoder = Encoder(
            n_src_vocab=vocab_size[0],  # 入力言語の語彙数
            max_len=args.max_len,  # 最大系列長
            n_layers=args.n_layer,  # レイヤー数
            n_head=args.n_head,  # ヘッド数
            d_k=args.d_key,  # keyの出力ベクトルの次元数
            d_v=args.d_value,  # valueの出力ベクトルの次元数
            d_model=args.d_model,  # 隠れ層の次元数
            d_inner_hid=args.d_inner_hid,  # Feed-Forwardの隠れ層2層目の次元数
            dropout=args.dropout,  # ドロップアウト率
            pad_idx=args.pad_idx,  # パディング用のID
        )

        self.decoder = Decoder(
            n_tgt_vocab=vocab_size[1],  # 出力言語の語彙数
            max_len=args.max_len,
            n_layers=args.n_layer,
            n_head=args.n_head,
            d_k=args.d_key,  # keyの出力ベクトルの次元数
            d_v=args.d_value,  # valueの出力ベクトルの次元数
            d_model=args.d_model,
            d_inner_hid=args.d_inner_hid,
            dropout=args.dropout,
            pad_idx=args.pad_idx,
            device=args.device,  # GPUを使うかどうか
        )

        self.tgt_word_proj = nn.Linear(args.d_model, vocab_size[1], bias=False)
        nn.init.xavier_normal_(self.tgt_word_proj.weight)
        self.device = args.device
        self.batch_size = args.batch_size
        self.bos_idx = args.bos_idx
        self.eos_idx = args.eos_idx
        self.max_len = args.max_len

        if args.proj_share_weight:
            # 出力言語の単語のEmbeddingと出力の写像で重みを共有
            self.tgt_word_proj.weight = self.decoder.tgt_word_emb.weight

    def forward(self, src: Tuple[Tensor, Tensor]) -> Tensor:
        """

        Args:
            src (Tuple[Tensor, Tensor]):

        Returns:
            predict_jp_sentence (str):

        """
        src_seq, src_pos = src
        src_seq = src_seq[:, 1:]  # <bos>を取り払う
        src_pos = src_pos[:, 1:]

        enc_output = self.encoder(src_seq, src_pos)

        tgt_seq = torch.full(
            [self.batch_size, 1], self.bos_idx, dtype=torch.long, device=self.device
        )
        tgt_pos = torch.arange(1, dtype=torch.long, device=self.device)
        tgt_pos = tgt_pos.unsqueeze(0).repeat(self.batch_size, 1)

        # 時刻ごとに処理
        for t in range(1, self.max_len + 1):
            print(f"--> {t} / {self.max_len} words")
            dec_output = self.decoder(tgt_seq, tgt_pos, src_seq, enc_output)  # type: ignore
            dec_output = self.tgt_word_proj(dec_output)  # type: ignore
            out = dec_output[:, -1, :].max(dim=-1)[1].unsqueeze(1)
            # 自身の出力を次の時刻の入力にする
            tgt_seq = torch.cat([tgt_seq, out], dim=-1)
            tgt_pos = torch.arange(t + 1, dtype=torch.long, device=self.device)
            tgt_pos = tgt_pos.unsqueeze(0).repeat(self.batch_size, 1)

            # release_gpu()       # 507s     58s

        result_seq = tgt_seq[:, 1:]

        return result_seq
