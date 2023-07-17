import collections
import os
import re
import csv

import tqdm
import torch
import spacy
import argparse
import torchtext
import pandas as pd
import numpy as np
import torchtext.transforms as T
import torchdata.datapipes as dp
import torchtext.functional as F

from typing import List, Tuple
from torchtext.vocab import vocab
from spacy.language import Language
from torch.utils.data import DataLoader
from collections import Counter, OrderedDict
from sklearn.model_selection import train_test_split
from torchtext.data.functional import generate_sp_model
from torchtext.data.functional import load_sp_model

# 自作クラス
from transformer_repro.datasets.custom_datasets import EnJpDataset
from transformer_repro.utils.utils import save_pkl, load_pkl


class CreateDataloader:
    """データローダーを作成するクラス"""

    def __init__(self, args: argparse.Namespace) -> None:
        self.device = args.device  # cpu or cuda
        self.batch_size = args.batch_size
        self.data_limit = args.data_limit  # データ制限数
        self.dataset_url = args.dataset_url
        self.pad_idx = args.pad_idx  # 0
        self.bos_idx = args.bos_idx  # 2
        self.eos_idx = args.eos_idx  # 3
        self.max_len = args.max_len  # 最大長
        self.special_token = args.special_token  # "<pad>","<unk>","<s>","</s>"

        self.csv_path = f"../data/dataset.csv"
        self.train_loader_name = f"../save/train_loader_{args.data_mode}_{str(args.data_limit)}_batch{str(args.batch_size)}_{str(args.device.type)}.pkl"
        self.valid_loader_name = f"../save/valid_loader_{args.data_mode}_{str(args.data_limit)}_batch{str(args.batch_size)}_{str(args.device.type)}.pkl"
        self.en_vocab_name = (
            f"../save/en_vocab_{args.data_mode}_{str(args.data_limit)}.pkl"
        )
        self.jp_vocab_name = (
            f"../save/jp_vocab_{args.data_mode}_{str(args.data_limit)}.pkl"
        )

        self.en_model_prefix = f"../data/sp/en_model_{args.data_limit}"
        self.jp_model_prefix = f"../data/sp/jp_model_{args.data_limit}"

        self.en_transform = None
        self.jp_transform = None
        self.pos_transform = T.Sequential(
            T.ToTensor(padding_value=self.pad_idx, dtype=torch.long),
            T.PadTransform(max_length=self.max_len, pad_value=self.pad_idx),
        )

    def _load_datasets(self) -> List[str]:
        """Web上にあるデータセットを読み込む

        Web上にあるtar.gzファイルを直接読み込み、正規化して返す

        Returns:
            raw_list (list): データセットの内容のリスト, 偶数インデックスに英語, 奇数インデックスに日本語の文が格納
        """
        # Web上にあるtar.gzファイルを読み込む
        datapipe = dp.iter.IterableWrapper([self.dataset_url])
        datapipe = datapipe.read_from_http()
        datapipe = datapipe.load_from_tar()

        _, stream = next(iter(datapipe))
        byte_data = stream.read()
        string_data = byte_data.decode("utf-8")
        raw_list = re.split(r"\n|\t", string_data)  # 正規化
        return raw_list

    def _generate_csv(self, raw_list: List[str]) -> None:
        """CSVファイルを作成

        正規化したデータセットをCSVファイルとして保存

        Args:
            raw_list (list):　データセットの内容のリスト, 偶数インデックスに英語, 奇数インデックスに日本語の文が格納
        """
        with open(self.csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["en", "jp"])
            # データの最大数
            num_rows = len(raw_list) // 2

            [
                writer.writerow([raw_list[i], raw_list[i + 1]])
                for i in range(0, num_rows, 2)
            ]
        print(f"generate file:{self.csv_path}")

    def _csv2df(self) -> pd.DataFrame:
        """CSVからデータフレームに変換

        CSVファイルを読み込み、指定されたデータ数のデータフレームを返す

        Returns:
            df (pd.DataFrame): 英語日本語を記述したデータフレーム, [0]:英語, [1]:日本語
        """
        df = pd.read_csv(self.csv_path)
        if self.data_limit:
            df = df[: self.data_limit]  # 制限
        return df

    def collate_batch(
        self, batch: List[Tuple[List[str], List[int], List[str], List[int]]]
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        """ミニバッチにまとめる

        単語分割された入力シーケンスをID化, テンソル化等をしてミニバッチにまとめる

        Args:
            batch (List[Tuple[List[str], List[int], List[str], List[int]]]): [英語シーケンス情報, 英語位置情報, 日本語シーケンス情報, 日本語位置情報], \
                            ※シーケンス情報は、単語分割されているだけの文章

        Returns:
            batch_x (Tuple[torch.Tensor, torch.Tensor]): 英語に関するバッチ　(シーケンス情報, 位置情報)
            batch_y (Tuple[torch.Tensor, torch.Tensor]): 日本語に関するバッチ　(シーケンス情報, 位置情報)

        Note:
            Dataloaderの出力としてミニバッチにまとめる,CreateDatasetクラスの__getitem__()で返される
        """
        src_seq, src_pos, tgt_seq, tgt_pos = zip(*batch)

        # バッチに対する前処理
        batch_x_seq = self.en_transform([seq for seq in src_seq]).to(self.device)  # type: ignore
        batch_y_seq = self.jp_transform([seq for seq in tgt_seq]).to(self.device)  # type: ignore

        # 単語の位置を表すテンソルを作成
        batch_x_pos = self.pos_transform([pos for pos in src_pos]).to(self.device)
        batch_y_pos = self.pos_transform([pos for pos in tgt_pos]).to(self.device)

        batch_x = (batch_x_seq, batch_x_pos)
        batch_y = (batch_y_seq, batch_y_pos)
        return batch_x, batch_y

    def _create_dataloader(self, dataset: EnJpDataset) -> DataLoader:
        """データローダーを生成

        Args:
            dataset (EnJpDataset): 英語日本語データセット

        Returns:
            dataloader (Dataloader): データローダー
        """
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collate_batch,
        )
        return dataloader

    def _load_pkl(self) -> Tuple[DataLoader, DataLoader, Tuple[int, int]]:
        """以前に読み込まれたデータセット情報を読み込む

        Returns:
            train_dataloader (DataLoader): 学習用データローダー
            valid_dataloader (DataLoader):　検証用データローダー
            vocab_size (tuple):　[英語語彙数, 日本語語彙数]
        """
        train_dataloader = load_pkl(self.train_loader_name)
        valid_dataloader = load_pkl(self.valid_loader_name)
        en_vocab = load_pkl(self.en_vocab_name)
        jp_vocab = load_pkl(self.jp_vocab_name)
        vocab_size = (len(en_vocab.vocab), len(jp_vocab))

        return train_dataloader, valid_dataloader, vocab_size

    def _save_pkl(
        self,
        train_dataloader: DataLoader,
        valid_dataloader: DataLoader,
        en_vocab: vocab,
        jp_vocab: vocab,
    ) -> None:
        """データセットをpklとして保存

        Args:
            train_dataloader (DataLoader): 学習用データローダー
            valid_dataloader (DataLoader): 検証用データローダー
            en_vocab (vocab): 英語語彙辞書
            jp_vocab (vocab): 日本語語彙辞書
        """
        save_pkl(self.train_loader_name, train_dataloader)
        save_pkl(self.valid_loader_name, valid_dataloader)
        save_pkl(self.en_vocab_name, en_vocab)
        save_pkl(self.jp_vocab_name, jp_vocab)

    def _create_transform(self, vocab_dic: vocab) -> torchtext.transforms.Sequential:
        """シーケンス変換処理を作成

        ID化, 最大長で切り捨て, 開始/終了トークンの追加, テンソル化, パディングを行う

        Args:
            vocab_dic (vocab): 英語, もしくは日本語の語彙辞書

        Returns:
            transform (torchtext.transforms.Sequential): 束ねた変換処理

        Note:
            ID化処理を行うため、英語、日本語の語彙が引数として必要となる
        """
        transform = T.Sequential(
            T.VocabTransform(vocab_dic),  # ID化
            T.Truncate(self.max_len - 2),  # 最大長以上のものは切り捨て      # args.max_len += 2　より
            T.AddToken(token=self.bos_idx, begin=True),  # 開始トークン追加
            T.AddToken(token=self.eos_idx, begin=False),  # 終了トークン追加
            T.ToTensor(padding_value=self.pad_idx, dtype=torch.long),  # テンソル化 & パディング
            T.PadTransform(max_length=self.max_len, pad_value=self.pad_idx),
        )
        return transform

    def raw2id(
        self, data_mode: str, en_vocab: vocab, predict_input: str
    ) -> torch.Tensor:
        """生の文章をID列に変換

        Args:
            data_mode (str): 'pretrain' or 'sp'
            en_vocab (vocab): 英語語彙辞書
            predict_input (str): 翻訳した英語文章

        Returns:
            id_seq (torch.Tensor): ID列
        """
        en_transform = self._create_transform(en_vocab)

        id_seq = torch.empty(0)  # 仮のテンソルを生成

        if data_mode == "pretrain":
            en_model = spacy.load("en_core_web_md")
            id_seq = en_transform([str(i) for i in en_model.tokenizer(predict_input)])
        else:
            en_sp_model = load_sp_model(self.en_model_prefix + ".model")
            id_seq = en_transform(en_sp_model.EncodeAsPieces(predict_input))

        return id_seq

    def raw2id_numpy(
        self, data_mode: str, en_vocab: vocab, predict_input: str
    ) -> np.ndarray:
        """生の文章をID列(numpy配列)に変換

        Args:
            data_mode (str): 'pretrain' or 'sp'
            en_vocab (vocab): 英語語彙辞書
            predict_input (str): 翻訳した英語文章

        Returns:
            id_seq (np.ndarray): ID列
        """

        # 単語分割
        if data_mode == "pretrain":
            en_model = spacy.load("en_core_web_md")
            id_seq = [
                str(i) for i in en_model.tokenizer(predict_input)
            ]  # ['What' ''s' 'up' '?']
        else:
            en_sp_model = load_sp_model(self.en_model_prefix + ".model")
            id_seq = en_sp_model.EncodeAsPieces(predict_input)

        # ID列化
        predict_id = T.VocabTransform(en_vocab)(id_seq)  # list

        # 最大長に合わせてカット
        if len(predict_id) > self.max_len - 2:
            predict_id = predict_id[: self.max_len - 2]

        # 開始/終了トークの付与, パディング
        predict_id = F.add_token(predict_id, self.bos_idx, begin=True)
        predict_id = F.add_token(predict_id, self.eos_idx, begin=False)
        if len(predict_id) - self.max_len != 0:
            insert_num = self.max_len - len(predict_id)
            predict_id = np.insert(
                predict_id, len(predict_id), [0] * insert_num
            )  # numpy化
        else:
            predict_id = np.array(predict_id)

        return predict_id


class CreatePretrainDataloader(CreateDataloader):
    """事前学習言語モデルを利用"""

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__(args)

    def __call__(self) -> Tuple[DataLoader, DataLoader, Tuple[int, int]]:
        """メイン処理

        Returns:
            train_dataloader (DataLoader): 学習用データローダー
            valid_dataloader (DataLoader): 検証用データローダー
            vocab_size (tuple[int, int]): [英語語彙辞書, 日本語語彙辞書]
        """
        if os.path.exists(self.valid_loader_name):
            train_dataloader, valid_dataloader, vocab_size = self._load_pkl()
            return train_dataloader, valid_dataloader, vocab_size

        if not os.path.exists(self.csv_path):
            self._generate_csv(self._load_datasets())

        df = self._csv2df()

        en_model = spacy.load("en_core_web_md")
        jp_model = spacy.load("ja_core_news_md")

        en_vocab = self._build_vocab(df["en"], en_model)
        jp_vocab = self._build_vocab(df["jp"], jp_model)
        vocab_size = (len(en_vocab.vocab), len(jp_vocab.vocab))

        self.en_transform = self._create_transform(en_vocab)
        self.jp_transform = self._create_transform(jp_vocab)

        train_df, valid_df = train_test_split(df, test_size=0.2)

        train_dataset = EnJpDataset(
            train_df, en_model.tokenizer, jp_model.tokenizer, self.max_len
        )
        valid_dataset = EnJpDataset(
            valid_df, en_model.tokenizer, jp_model.tokenizer, self.max_len
        )

        train_dataloader = self._create_dataloader(train_dataset)
        valid_dataloader = self._create_dataloader(valid_dataset)

        self._save_pkl(train_dataloader, valid_dataloader, en_vocab, jp_vocab)

        return train_dataloader, valid_dataloader, vocab_size

    def _build_vocab(self, df: pd.DataFrame, model: Language) -> vocab:
        """語彙辞書を作成

        Args:
            df (pd.DataFrame): 英語,もしくは日本語のデータフレーム
            model (spacy.language): 事前学習済みの言語モデル

        Returns:
            vocab_dic (vocab): 英語, もしくは日本語の語彙辞書
        """

        def tokenizer(sentence):
            return [tok.text for tok in model.tokenizer(sentence)]

        counter: collections.Counter = Counter()
        [counter.update(tokenizer(df[i])) for i in tqdm.tqdm(range(len(df)))]
        vocab_dic = vocab(counter, min_freq=2, specials=self.special_token)
        vocab_dic.set_default_index(vocab_dic["<unk>"])
        return vocab_dic


class CreateSpDataloader(CreateDataloader):
    """Sentence Pieceを利用"""

    def __init__(self, args):
        super().__init__(args)

        os.makedirs("./data/sp", exist_ok=True)

        self.en_text_path = f"./data/sp/en_{args.data_limit}.txt"
        self.jp_text_path = f"./data/sp/jp_{args.data_limit}.txt"

        self.en_vocab_path = f"./data/sp/en_model_{args.data_limit}.vocab"
        self.jp_vocab_path = f"./data/sp/jp_model_{args.data_limit}.vocab"

    def __call__(self):
        """メイン処理

        Returns:
            train_dataloader
            valid_dataloader
            vocab_size

        Note:
            spの場合、エラーが出るため,pklによる保存はしない
        """
        if not os.path.exists(self.csv_path):
            self._generate_csv(self._load_datasets())

        df = self._csv2df()

        en_sp_model = self._generate_sp_model(
            df["en"], self.en_model_prefix, self.en_text_path
        )
        jp_sp_model = self._generate_sp_model(
            df["jp"], self.jp_model_prefix, self.jp_text_path
        )

        en_vocab = self._build_vocab(self.en_vocab_path)
        jp_vocab = self._build_vocab(self.jp_vocab_path)
        vocab_size = (len(en_vocab.vocab), len(jp_vocab.vocab))

        self.en_transform = self._create_transform(en_vocab)
        self.jp_transform = self._create_transform(jp_vocab)

        train_df, valid_df = train_test_split(df, test_size=0.2)

        train_dataset = EnJpDataset(
            train_df, en_sp_model.EncodeAsPieces, jp_sp_model.EncodeAsPieces
        )
        valid_dataset = EnJpDataset(
            valid_df, en_sp_model.EncodeAsPieces, jp_sp_model.EncodeAsPieces
        )

        train_dataloader = self._create_dataloader(train_dataset)
        valid_dataloader = self._create_dataloader(valid_dataset)

        return train_dataloader, valid_dataloader, vocab_size

    @staticmethod
    def _generate_sp_model(df, model_prefix, text_path):
        """spモデルを作成

        Args:
            df
            model_prefix
            text_path

        Returns:
            sp_model
        """
        if not os.path.exists(model_prefix + ".model"):
            with open(text_path, "w", encoding="utf-8") as w:
                w.write("\n".join(df.tolist()))
            # 学習
            generate_sp_model(text_path, model_prefix=model_prefix)

        sp_model = load_sp_model(model_prefix + ".model")
        return sp_model

    @staticmethod
    def _tokens_seq(file):
        """

        Args:
            file

        Returns:

        """
        with open(file, encoding="utf-8") as f:
            file_data = f.read()
        return [s.split("\t")[0] for s in file_data.split("\n")]

    def _build_vocab(self, vocab_path: str):
        """語彙辞書作成

        Args:
            vocab_path (str):

        Returns:
            vocab_dic ():
        """
        vocab_dic = vocab(
            OrderedDict([(token, 1) for token in self._tokens_seq(vocab_path)]),
            specials=self.special_token,
            special_first=True,
        )
        vocab_dic.set_default_index(vocab_dic["<unk>"])
        return vocab_dic
