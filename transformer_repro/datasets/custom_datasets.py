import pandas as pd
import spacy.tokenizer

from torch.utils.data import Dataset


class EnJpDataset(Dataset):
    """英語-日本語の翻訳データセット

    カスタムデータセット: Datasetクラスを継承
    """

    def __init__(
        self,
        df: pd.DataFrame,
        en_tokenizer: spacy.tokenizer.Tokenizer,
        jp_tokenizer: spacy.tokenizer.Tokenizer,
        max_len: int,
    ) -> None:
        """

        Args:
            df (pd.DataFrame): 英語日本語データフレーム ['en']: 英語データ, ['jp']: 日本語データ
            en_tokenizer (spacy.tokenizer.Tokenizer): 英語トークナイザ
            jp_tokenizer (spacy.tokenizer.Tokenizer): 日本語トークナイザ
            max_len (int): 最大長

        """
        self.df = df
        self.src = df["en"]
        self.tgt = df["jp"]

        self.max_len = max_len

        self.en_tokenizer = en_tokenizer
        self.jp_tokenizer = jp_tokenizer

    @staticmethod
    def tokenizer(tokenizer, sentence):
        return [str(tok) for tok in tokenizer(sentence)]

    def __getitem__(self, index):
        """学習時に呼び出される関数

        Args:
            index (int): 指定インデックス

        Returns:
            src_text (list): 英語列
            tgt_text (list): 日本語列

        Note:
            datasets_main.pyのcollate_batch()にReturnsが渡る、バッチ数だけ__getitem__関数が呼び出される
        """

        src_text = self.src.iloc[index]  # ex) here's something good.
        tgt_text = self.tgt.iloc[index]  # ex) いいものがあるぞ

        # 前処理（単語分割）
        src_text = self.tokenizer(
            self.en_tokenizer, src_text
        )  # ex) 'here', 's', 'something', 'good', '.'
        tgt_text = self.tokenizer(
            self.jp_tokenizer, tgt_text
        )  # ex) 'いい', 'もの', 'が', 'ある', 'ぞ'

        # 位置ベクトルを作成
        src_len = min(len(src_text) + 2, self.max_len)  # <BOS>,<EOS>の分があるため、+2が必要
        tgt_len = min(
            len(tgt_text) + 2, self.max_len
        )  # 打ち切られる文章があるため、max_lenを超えないようにmin()
        src_pos = list(range(1, src_len + 1))
        tgt_pos = list(range(1, tgt_len + 1))
        return src_text, src_pos, tgt_text, tgt_pos

    def __len__(self):
        return len(self.df)
