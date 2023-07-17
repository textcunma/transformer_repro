import time
import torch
import GPUtil
import pickle
import datetime
import numpy as np
import torch.nn as nn

from pynvml import *

from torch import Tensor
from nltk.translate import bleu_score

# from torchtext.vocab import Vocab
from typing import Any, Dict, List
from torch.utils.data.dataloader import DataLoader


def save_pkl(name: str, file: DataLoader) -> None:
    """pklファイルとして保存

    Args:
        name (str): 保存ファイル名
        file (Any): 保存データ
    """
    with open(name, "wb") as f:
        pickle.dump([file], f)


def load_pkl(name: str) -> Any:
    """pklファイルを読み込み

    Args:
        name (str): ファイル名

    Returns:
        file (Any): 保存されていたデータ
    """
    with open(name, "rb") as f:
        file = pickle.load(f)[0]
    return file


def update_values(dict_from: Dict[str, Any], dict_to: Dict[str, Any]) -> None:
    """引数をymlファイルの内容に更新

    Args:
        dict_from (Dict): ymlファイルに記述された新規引数
        dict_to (Dict): 更新される側の引数

    Note:
        Ref: https://github.com/salesforce/densecap/blob/5d08369ffdcb7db946ae11a8e9c8a056e47d28c2/data/utils.py#L85
    """
    for key, value in dict_from.items():
        if isinstance(value, dict):
            update_values(dict_from[key], dict_to[key])
        elif value is not None:
            dict_to[key] = dict_from[key]


def calc_bleu(refs: List[List[int]], hyps: List[List[int]], eos_idx: int) -> float:
    """BLEUスコアを計算する関数

    Args:
        refs (list): 正解列
        hyps (list): 予測列
        eos_idx (int): EOSインデックス

    Returns:
        bleu (float): BLEUスコア(0~100)

    Note:
        refs: List[List[Any]]とするとmypyでエラーが消えるが、実際List[List[int]]で良いと考えている
    """
    refs = [[ref[: ref.index(eos_idx)]] for ref in refs]  # type: ignore
    hyps = [hyp[: hyp.index(eos_idx)] if eos_idx in hyp else hyp for hyp in hyps]
    bleu = 100 * bleu_score.corpus_bleu(
        refs, hyps, smoothing_function=bleu_score.SmoothingFunction().method1
    )
    return bleu  # type: ignore


def get_attn_padding_mask(seq_q: Tensor, seq_k: Tensor, pad_idx: int) -> Tensor:
    """Paddingトークンにattentionをさせないためのマスクを作成

    各バッチ内の各 seq_q 要素に対して、seq_k の対応する要素がパディングされているかどうかを示すパディングマスク

    Args:
        seq_q (Tensor): queryの系列, shape [batch_size, len_q]
        seq_k (Tensor): keyの系列, shape [batch_size, len_k],
        pad_idx (int): paddingのインデックス

    Returns:
        pad_attn_mask (Tensor): shape [batch_size, len_q, len_k]

    Examples:
        seq_q: torch.tensor [1,2,3]
        seq_k: torch.tensor [4,5,6,PAD]

        pad_attn_mask: torch.tensor [[False, False, False, False, True],   <- True部分は, 「1」と「PAD」との間を注視しないようにする
                                    [False, False, False, False, True],    <- True部分は, 「2」と「PAD」との間を注視しないようにする
                                    [False, False, False, False, True]]    <- True部分は, 「3」と「PAD」との間を注視しないようにする

    """
    batch_size, len_q = seq_q.size()
    len_k = seq_k.size()[1]
    pad_attn_mask = seq_k.data.eq(pad_idx).unsqueeze(1)  # (N, 1, len_k)
    pad_attn_mask = pad_attn_mask.expand(batch_size, len_q, len_k)  # (N, len_q, len_k)
    return pad_attn_mask


def get_attn_subsequent_mask(seq: Tensor, device: torch.device) -> Tensor:
    """未来情報カンニング防止のためにattentionを0にするためのマスクを作成

    Args:
        seq (Tensor): shape [batch_size, length]
        device (torch.device): cuda or cpu

    Returns:
        subsequent_mask (Tensor): shape [batch_size, length, length]
    """
    attn_shape = (seq.size(1), seq.size(1))
    # 上三角行列(diagonal=1: 対角線より上が1で下が0)
    subsequent_mask = torch.triu(
        torch.ones(attn_shape, dtype=torch.uint8, device=device), diagonal=1
    )
    subsequent_mask = subsequent_mask.repeat(seq.size(0), 1, 1)
    return subsequent_mask


# def ids_to_sentence(vocab: Vocab, ids: List[int]) -> str:
def ids_to_sentence(vocab, ids: List[int]) -> str:
    """IDリストを単語リストに変換

    Args:
         vocab (Vocab): 語彙辞書
         ids (List[int]): IDリスト

    Returns:
         seq (str): 単語リスト
    """
    seq = "".join([vocab.vocab.itos_[_id] for _id in ids])
    return seq


def trim_eos(ids: List[int], eos_id: int) -> List[int]:
    """IDリストからEOS以降の単語を除外

    Args:
         ids (List[int]): IDリスト
         eos_id (int): EOSのID

    Returns:
         ids (List[int]): IDリスト
    """
    return ids[: ids.index(eos_id)] if eos_id in ids else ids


def position_encoding_init(n_pos: int, d_model: int) -> torch.Tensor:
    """位置エンコーディングのための位置埋め込み行列の初期化

    Args:
        n_pos (int): 位置数  ->  max_len + 1
        d_model (int): 隠れ次元数

    Returns:
        pos_tensor (torch.Tensor): 位置埋め込みテンソル
    """
    # PADがある単語の位置はpos=0にしておき、position_encも0にする
    position_enc = np.array(
        [
            [pos / np.power(10000, 2 * (j // 2) / d_model) for j in range(d_model)]
            if pos != 0
            else np.zeros(d_model)
            for pos in range(n_pos)
        ]
    )

    position_enc[1:, 0::2] = np.sin(
        position_enc[1:, 0::2]
    )  # dim 2i      0::2とは、0,2,4,6,8,10,...という意味
    position_enc[1:, 1::2] = np.cos(
        position_enc[1:, 1::2]
    )  # dim 2i+1    1::2とは、1,3,5,7,9,/....という意味

    pos_tensor = torch.tensor(position_enc, dtype=torch.float)
    return pos_tensor


def set_gpu(device: torch.device, model: nn.Module):
    """GPUをセット

    Args:
        device (torch.device): cpu or cuda
        model (nn.Module): Transformerモデル
    """
    if device.type == "cuda":
        model = torch.nn.DataParallel(model) if torch.cuda.device_count() > 1 else model
        model.to(device)


def print_time(func):
    """デコレーター：ログ出力"""

    def wrapper(*args, **kargs):
        start = time.time()
        output = func(*args, **kargs)
        end = time.time()
        print(f"処理時間：{str(round(end - start, 2))}[s]")
        return output

    return wrapper

def decide_csv_name() -> str:
    """CSVファイルの名前を決定

    CSVファイルの名前は、実行した日時と時間の組み合わせとする

    Returns:
        csv_name (str): CSVファイルの名前
    """
    dt = datetime.datetime.now()
    csv_name = "../csv_log/{}月{}日{}時{}分_log.csv".format(
        dt.month, dt.day, dt.hour, dt.minute
    )
    return csv_name

def calc_time(func):
    """デコレーター：ログ出力"""

    def wrapper(*args, **kargs):
        start = time.time()
        output = func(*args, **kargs)
        end = time.time()
        elapsed_time = end - start
        return output, elapsed_time

    return wrapper


def release_gpu():
    torch.cuda.empty_cache()
    GPUtil.showUtilization()


def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used // 1024 ** 2} MB.")