import torch
import argparse

# 自作関数、クラス
from .base import BaseProcess
from transformer_repro.datasets.custom_dataloader import (
    CreateSpDataloader,
    CreateDataloader,
)
from transformer_repro.utils.utils import ids_to_sentence, trim_eos, print_time, load_pkl


class PredictProcess(BaseProcess):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__(args)

        self.bos_idx = args.bos_idx
        self.data_mode = args.data_mode
        self.cd = CreateDataloader(args)
        self.csd = CreateSpDataloader(args)

        self.en_vocab, self.jp_vocab = self._load_vocab()
        vocab_size = (len(self.en_vocab.vocab), len(self.jp_vocab.vocab))
        self.model = self._init_model(args, vocab_size, is_train=False, jp_vocab=self.jp_vocab)

    @print_time
    def __call__(self, predict_input: str = "What's up?") -> None:
        """メイン実行

        Args:
            predict_input (str): 翻訳したい英文

        """
        src_seq = (
            self.cd
            .raw2id(self.data_mode, self.en_vocab, predict_input)
            .to(self.device)
        )
        zero_index = torch.argmin(src_seq)
        src_pos = torch.tensor(
            list(range(1, zero_index + 1)) + [0] * (self.max_len - zero_index),
            dtype=torch.long,
            device=self.device,
        )

        src_seq = torch.unsqueeze(src_seq, dim=0)
        src_pos = torch.unsqueeze(src_pos, dim=0)

        # 文章を予測
        self.model.eval()
        with torch.autocast(device_type=self.device.type, enabled=self.use_amp,
                            dtype=self.amp_dtype):
            with torch.no_grad():
                result_seq = self.model((src_seq, src_pos))

        pred_ids = result_seq[0].data.cpu().tolist()
        predict_jp_sentence = ids_to_sentence(
            self.jp_vocab, trim_eos(pred_ids, self.eos_idx)
        )

        print(f"src:{predict_input}")
        print(f"out:{predict_jp_sentence}")

    def _load_vocab(self):
        """語彙情報を読み込む"""
        if self.data_mode == 'pretrain':
            en_vocab_name = f"../save/en_vocab_{self.data_mode}_{str(self.data_limit)}.pkl"
            jp_vocab_name = f"../save/jp_vocab_{self.data_mode}_{str(self.data_limit)}.pkl"
            en_vocab = load_pkl(en_vocab_name)
            jp_vocab = load_pkl(jp_vocab_name)
        else:
            _, _, en_vocab, jp_vocab = self.csd()   # Sentence Pieceの場合

        return en_vocab, jp_vocab
