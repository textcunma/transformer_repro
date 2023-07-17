import torch
import random
import argparse
import torch_ort
import torch.nn as nn
import torch.optim as optim

from typing import Tuple

from torch.utils.data import DataLoader
from transformer_repro.utils.utils import set_gpu
from transformer_repro.model.transformer import Transformer, TransformerPredict
from transformer_repro.datasets.custom_dataloader import CreateSpDataloader
from transformer_repro.datasets.custom_dataloader import CreatePretrainDataloader


class BaseProcess:
    """学習,テスト,予測のベースとなるクラス"""

    def __init__(self, args: argparse.Namespace) -> None:
        self.device = args.device  # 'cpu' or 'cuda'
        self.data_mode = args.data_mode  # 'pretrain' or 'sp'
        self.use_amp = args.use_amp
        self.data_limit = args.data_limit  # データ数制限
        self.batch_size = args.batch_size  # バッチサイズ
        self.max_len = args.max_len  # 最大長
        self.eos_idx = args.eos_idx  # 3
        self.checkpoint_dir = args.checkpoint_dir

        self.optimizer = None
        self.scheduler = None

        if self.device.type == 'cpu':
            self.amp_dtype = torch.bfloat16  # CPUでは'bfloat16'のみしか対応していない
        else:
            self.amp_dtype = torch.float16

    def _init_model(self, args: argparse.Namespace, vocab_size: tuple[int, int], is_train: bool, jp_vocab = None) -> nn.Module:
        """モデルの初期化, オプティマイザ, スケジューラーの設定

        Args:
            args (argparse.Namespace): コマンドライン引数
            vocab_size (tuple): [英語の語彙数, 日本語の語彙数]
            is_train (bool): 学習の場合、True
            jp_vocab (vocab): 日本語語彙

        Returns:
            model (Transformer): Transformerモデル
        """

        if is_train:
            model = Transformer(args, vocab_size)
        else:
            model = TransformerPredict(args, vocab_size)

        if is_train:
            # Position Encoding部分のパラメータを固定(学習させない)
            freeze_param = ["encoder.position_enc.weight", "decoder.position_enc.weight"]
            for name, param in model.named_parameters():
                if name in freeze_param:
                    param.requires_grad = False

            # オプティマイザ等を設定
            self.optimizer = optim.Adam(model.parameters(), lr=args.lr)
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, "min")

        # 学習済みの重みを読み込む
        if args.use_pretrain:
            self._update_model_pretrain(model, is_train)

        # GPUの使用設定
        set_gpu(args.device, model)  # type: ignore

        # PyTorch 2.0からの新機能: torch.compile()  (cuda 11.7以上, windowsは不可)
        try:
            # model = torch.compile(model, backend="inductor")
            model = torch.compile(model)
        except RuntimeError as e:
            print(e)
            print("Alternatively, use 'torch_ort' ")
            model = torch_ort.ORTModule(model)

        return model

    def _update_model_pretrain(self, model, is_train: bool):
        """学習済みデータでモデルを更新

        Notes:
            参考: https://qiita.com/NLPingu/items/461bfcf54344e6f5da33
        """
        try:
            ckpt = torch.load("../save/ckpt.bin")
            ckpt_model = self._refine_load_model(ckpt['model'])

            if hasattr(model, "module"):  # DataParallelを使用した場合
                model.module.load_state_dict(ckpt_model)
            else:
                model.load_state_dict(ckpt_model)

            if is_train:
                self.optimizer.load_state_dict(ckpt["optimizer"])
                self.scheduler.load_state_dict(ckpt["scheduler"])

            random.setstate(ckpt["random"])
            # np.random.set_state(ckpt["np_random"])
            torch.set_rng_state(ckpt["torch"])
            torch.random.set_rng_state(ckpt["torch_random"])
            torch.cuda.set_rng_state(ckpt["cuda_random"])  # gpuを使用する場合は必要
            torch.cuda.torch.cuda.set_rng_state_all(ckpt["cuda_random_all"])  # 複数gpuを使用する場合は必要

        except FileNotFoundError as e:
            print(e)
            raise "the pretrained model don't exist"
        except RuntimeError as e:
            print(e)
            raise "Error: load_state_dict"

    @staticmethod
    def _refine_load_model(ckpt_model):
        """Pytorch compile機能を使った場合への調整
        Notes:
            参考: https://crispy-data.com/post/4059fdab-7a91-44d0-9ab2-8dd17c5db288
        """
        restored_ckpt = {}
        for k, v in list(ckpt_model.items()):
            restored_ckpt[k.replace('_orig_mod.', '')] = v

        return restored_ckpt

    @staticmethod
    def _build_dataloader(
            args: argparse.Namespace,
    ) -> Tuple[DataLoader, DataLoader, Tuple[int, int]]:
        """データローダ作成

        Args:
            args (argparse.Namespace): コマンドライン引数

        Returns:
            train_dataloader (DataLoader): 学習用データローダー
            valid_dataloader (DataLoader): 検証用データローダー
            vocab_size (Tuple[int, int]): [英語語彙数, 日本語語彙数]
        """
        if args.data_mode == "pretrain":
            train_dataloader, valid_dataloader, vocab_size = CreatePretrainDataloader(
                args
            )()
        elif args.data_mode == "sp":
            train_dataloader, valid_dataloader, vocab_size = CreateSpDataloader(args)()
        else:
            raise Exception("The mode doesn't exist")

        return train_dataloader, valid_dataloader, vocab_size
