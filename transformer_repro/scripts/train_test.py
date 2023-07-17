import csv
import torch
import random
import argparse
import torch.nn as nn

from typing import Tuple, List
from dataclasses import dataclass


# 自作関数・クラス
from .base import BaseProcess
from transformer_repro.utils.show_visdom import ShowVisdom
from transformer_repro.utils.utils import (
    calc_bleu,
    calc_time,
    decide_csv_name,
)


@dataclass
class LearnInfo:
    train_loss: float  # 損失値(学習時)
    train_refs: list  # 正解データ(学習時)
    train_hyps: list  # 予測データ(学習時)
    train_bleu: float  # BLEUスコア(学習時)
    valid_loss: float  # 損失値(検証時)
    valid_refs: list  # 正解データ(検証時)
    valid_hyps: list  # 予測データ(検証時)
    valid_bleu: float  # BLEUスコア(検証時)


class TrainTestProcess(BaseProcess):
    """深層学習の学習/テストプロセス"""

    def __init__(self, args: argparse.Namespace) -> None:
        """
        Notes:
            参考：https://tawara.hatenablog.com/entry/2021/05/31/220936ｂ
        """
        super().__init__(args)
        self.train_dataloader, self.valid_dataloader, self.vocab_size = self._build_dataloader(args)
        self.n_batches = len(self.train_dataloader)  # ミニバッチ数
        self.lr = args.lr  # 学習率
        self.epochs = args.epochs  # エポック数

        self.model = self._init_model(args, self.vocab_size, is_train=True)
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=args.pad_idx, reduction="sum"
        ).to(args.device)

        if self.device.type == 'cuda':
            self.scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)  # 初期化時にAMPを使うかを渡す

        self.use_visdom = args.use_visdom
        if args.use_visdom:
            self.show_vis = ShowVisdom()

        self.csv_name = decide_csv_name()
        self._create_csv()

    def __call__(self) -> None:
        """メイン処理"""
        best_valid_bleu = 0.0

        for epoch in range(self.epochs):
            lr_info, elapsed_time = self._epoch_process(epoch)

            # 検証データでBLEUスコアが改善した場合、モデルを保存
            if lr_info.valid_bleu > best_valid_bleu:
                self._save_model()
                best_valid_bleu = lr_info.valid_bleu

            # release_gpu()
            if self.use_visdom:
                self.show_vis.train_valid(lr_info.train_loss, lr_info.valid_loss)
                self.show_vis.bleu(lr_info.train_bleu, lr_info.valid_bleu)
            else:
                self._print_lr_info(epoch, elapsed_time, lr_info)

            self._add_csv(epoch, elapsed_time, lr_info)

        # Visdom終了
        if self.use_visdom:
            self.show_vis.__exit__()

    @calc_time
    def _epoch_process(self, epoch):
        """エポックごとの処理

        学習, 検証を行い, 損失, BLEUスコア等を記録した'lr_info'を返す

        Args:
            epoch (int): 現在のエポック数

        Returns:
            lr_info (LearnInfo): 学習情報

        """
        lr_info = LearnInfo(0.0, [], [], 0.0, 0.0, [], [], 0.0)

        self._train(lr_info, epoch)  # 学習
        self._valid(lr_info)  # 検証

        # 損失をサンプル数で割って正規化
        lr_info.train_loss /= float(self.train_dataloader.sampler.num_samples)  # type: ignore
        lr_info.valid_loss /= float(self.valid_dataloader.sampler.num_samples)  # type: ignore

        # 学習率の動的変更
        self.scheduler.step(lr_info.valid_loss)

        # BLEU(0~100)を計算
        lr_info.train_bleu = calc_bleu(
            lr_info.train_refs, lr_info.train_hyps, self.eos_idx
        )
        lr_info.valid_bleu = calc_bleu(
            lr_info.valid_refs, lr_info.valid_hyps, self.eos_idx
        )
        return lr_info

    def _train(self, lr_info: LearnInfo, epoch: int) -> None:
        """学習工程

        Args:
            lr_info (LearnInfo): 学習に使用する損失値等のデータクラス
            epoch (int): エポック数

        Notes:
            enumerate関数が動作すると、EnJpDatasetクラスの__getitem__()が動作
        """
        for train_iter, (batch_X, batch_Y) in enumerate(self.train_dataloader):
            loss, gt, pred = self._compute_loss(batch_X, batch_Y, is_train=True)
            lr_info.train_loss += loss
            lr_info.train_refs += gt
            lr_info.train_hyps += pred

            # visdomによる可視化(イテレーション単位)
            if self.use_visdom:
                self.show_vis.train_itr(loss, epoch, self.n_batches, train_iter)

    def _valid(self, lr_info: LearnInfo):
        """検証工程

        Args:
            lr_info (LearnInfo): 学習に使用する損失値等のデータクラス
        """
        for batch_x, batch_y in self.valid_dataloader:
            loss, gt, pred = self._compute_loss(batch_x, batch_y, is_train=False)
            lr_info.valid_loss += loss
            lr_info.valid_refs += gt
            lr_info.valid_hyps += pred

    def test(self):
        """テスト用, BLEUスコアを計算(外部参照可能)

        Returns:
            bleu_score (float): BLEUスコア

        """
        lr_info = LearnInfo(0.0, [], [], 0.0, 0.0, [], [], 0.0)
        self._valid(lr_info)
        bleu_score = calc_bleu(lr_info.valid_refs, lr_info.valid_hyps, self.eos_idx)
        print(f"BLEU score: {bleu_score}")

    def _compute_loss(
            self,
            batch_x: Tuple[torch.Tensor, torch.Tensor],
            batch_y: Tuple[torch.Tensor, torch.Tensor],
            is_train: bool,
    ) -> Tuple[float, List[List[int]], List[List[int]]]:
        """バッチの損失を計算

        Args:
            batch_x (tuple[torch.Tensor, torch.Tensor]): 英語に関するバッチ(シーケンス情報, 位置情報), 各情報は(batch_size, max_length + 2)
            batch_y (tuple[torch.Tensor, torch.Tensor]): 日本語に関するバッチ(シーケンス情報, 位置情報), 各情報は(batch_size, max_length + 2)
            is_train (bool): Trueの場合は誤差逆伝播を行う

        Returns:
            loss (float): 損失値
            gt (list): 正解データ len = batch_size*(max_length-1), 中身はID列, len() = バッチサイズ
            pred (list): 予測データ len = batch_size*(max_length-1), 中身はID列, len() = バッチサイズ
        """
        # モデルを学習モード or 推論モードにする
        self.model.train(is_train)

        # 予測、正解値、損失を計算,学習時のみ勾配を計算
        with torch.set_grad_enabled(is_train):
            # 順方向の計算
            with torch.autocast(device_type=self.device.type, enabled=self.use_amp, dtype=self.amp_dtype):
                pred_y = self.model(batch_x, batch_y)
                gt = batch_y[0][:, 1:].contiguous()
                loss = self.criterion(pred_y.view(-1, pred_y.size(2)), gt.view(-1))

        # 学習時は誤差逆伝播によるパラメータの更新
        if is_train:
            # 重みの更新(誤差逆伝播計算)を始める前に明示的に勾配を初期化する必要
            self.optimizer.zero_grad()  # 勾配の初期化
            if self.device.type == 'cuda':
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()  # 勾配計算(逆方向の計算)
                self.optimizer.step()  # パラメーター更新

        gt = gt.data.cpu().numpy().tolist()
        pred = pred_y.max(dim=-1)[1].data.cpu().numpy().tolist()
        loss = loss.item()

        return loss, gt, pred  # type: ignore

    def _save_model(self) -> None:
        """モデルを保存
        Notes:
            参考: https://qiita.com/NLPingu/items/461bfcf54344e6f5da33
        """
        # DataParallelを使用している場合はmodel.moduleを取り出す
        model_to_save = self.model.module if hasattr(self.model, "module") else self.model
        ckpt = {
            "model": model_to_save.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "random": random.getstate(),
            # "np_random": np.random.get_state(),  # numpy.randomを使用する場合は必要
            "torch": torch.get_rng_state(),
            "torch_random": torch.random.get_rng_state(),
            "cuda_random": torch.cuda.get_rng_state(),  # gpuを使用する場合は必要
            "cuda_random_all": torch.cuda.get_rng_state_all(),  # 複数gpuを使用する場合は必要
        }

        torch.save(ckpt, "../save/checkpoint.bin")

    def _create_csv(self) -> None:
        """学習結果を記録するためにCSVに記述"""
        with open(self.csv_name, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "データモード",
                    self.data_mode,
                    "データ数",
                    self.data_limit,
                    "batch",
                    self.batch_size,
                ]
            )
            writer.writerow(
                [
                    "max_len",
                    self.max_len,
                    "学習率",
                    self.lr,
                    "optimizer",
                    str(type(self.optimizer)),
                ]
            )
            writer.writerow(
                [
                    "epoch",
                    "time",
                    "train_loss",
                    "train_bleu",
                    "valid_loss",
                    "valid_bleu",
                ]
            )

    def _add_csv(self, epoch: int, elapsed_time: float, lr_info: LearnInfo) -> None:
        """エポックごとの学習結果を記録するために追記

        Args:
            epoch (int): エポック数
            elapsed_time (float): １エポックにかかった時間
            lr_info (LearnInfo): 学習時の情報を記録するためのデータクラス
        """

        with open(self.csv_name, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    epoch,
                    elapsed_time,
                    lr_info.train_loss,
                    lr_info.train_bleu,
                    lr_info.valid_loss,
                    lr_info.valid_bleu,
                ]
            )

    @staticmethod
    def _print_lr_info(epoch: int, elapsed_time: float, lr_info: LearnInfo) -> None:
        """エポックごとの学習結果を標準出力

        Args:
            epoch (int): エポック数
            elapsed_time (float): １エポックにかかった時間
            lr_info (LearnInfo): 学習時の情報を記録するためのデータクラス
        """
        epoch_info = "Epoch {} [{:.1f}s]:train_loss:{:5.2f} train_bleu:{:2.2f} valid_loss:{:5.2f} valid_bleu:{:2.2f}"
        epoch_info.format(
            epoch + 1,
            elapsed_time,
            lr_info.train_loss,
            lr_info.train_bleu,
            lr_info.valid_loss,
            lr_info.valid_bleu,
        )
        print(epoch_info)
        print("-" * 80)
