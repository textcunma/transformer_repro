import os
import yaml
import torch
import argparse

# 自作関数,クラス
from transformer_repro.scripts.predict import PredictProcess
from transformer_repro.scripts.train_test import TrainTestProcess
from transformer_repro.utils.utils import update_values
from transformer_repro.scripts.onnx_predict import OnnxPredictProcess
from transformer_repro.scripts.oputuna import OptunaProcess


def main(args: argparse.Namespace) -> None:
    # 行列計算の精度を設定(https://sebastianraschka.com/blog/2023/llm-mixed-precision-copy.html)
    torch.set_float32_matmul_precision('high')  # highest, high, medium

    os.makedirs("../data", exist_ok=True)  # データセット保存フォルダの作成
    os.makedirs("../csv_log", exist_ok=True)  # 学習状況の記録用フォルダの作成
    os.makedirs("../save", exist_ok=True)  # データローダ, モデルのパラメータの保存用フォルダの作成

    if args.mode == "train":
        print("start train process")
        TrainTestProcess(args)()

    elif args.mode == "test":
        print("start test process")
        TrainTestProcess(args).test()

    elif args.mode == "predict":
        print("start prediction process")
        PredictProcess(args)(predict_input=args.predict_input)

    elif args.mode == "onnx":
        print("start prediction process - ONNX version")
        OnnxPredictProcess(args)(predict_input=args.predict_input)

    elif args.mode == "optuna":
        print("start hyper parameter tuning")
        OptunaProcess(args)()

    else:
        raise Exception("The mode doesn't exist")


def refine_args(args: argparse.Namespace) -> argparse.Namespace:
    """与えられた引数を調整"""

    # <BOS>と<EOS>を含めた最大系列長
    args.max_len += 2

    # ymlファイルに記述された引数で更新
    with open("../cfg.yml", "r", encoding="utf-8") as handle:
        options_yaml = yaml.load(handle, Loader=yaml.SafeLoader)
    update_values(options_yaml, vars(args))

    # cudaが使用できるならば、cudaに設定
    if args.device == "cpu":
        args.device = torch.device("cpu")
    else:
        args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # データ数の制限
    if args.data_limit and type(args.data_limit) == str:
        args.data_limit = eval(args.data_limit)

    # データ数の少なさが原因によるSPでのエラー回避
    if args.data_mode == "sp" and args.data_limit < 10**6:
        raise Exception("When sp mode, Need data_limit is more than 10**6")

    # 学習時以外はvisdomは使用しない
    if args.use_visdom and args.mode != "train":
        args.use_visdom = False

    # test, predictの場合, 学習済みの重みを必ず使用
    if args.mode == 'test' or args.mode == 'predict':
        args.use_pretrain = True

    return args


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="コマンドライン引数")

    #
    # モード設定
    #
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "test", "predict", "onnx", "optuna"],
        default="train",
        help="学習モード or テストモード or 予測モード or ONNXによる予測モード or Optunaによるハイパラ最適化",
    )
    parser.add_argument(
        "--data_mode",
        type=str,
        choices=["pretrain", "sp"],
        default="pretrain",
        help="事前学習済み言語モデル or SentencePiece",
    )
    #
    # データセット関連
    #
    parser.add_argument(
        "--dataset_url",
        type=str,
        default="https://nlp.stanford.edu/projects/jesc/data/raw.tar.gz",
        help="データセットのURL",
    )
    parser.add_argument(
        "--special_token",
        default=("<pad>", "<unk>", "<s>", "</s>"),
        help="特殊トークン",
    )
    parser.add_argument(
        "--pad_idx",
        default=0,
        help="パディングトークンのインデックス",
    )
    parser.add_argument(
        "--unk_idx",
        default=1,
        help="未知語トークンのインデックス",
    )
    parser.add_argument(
        "--bos_idx",
        default=2,
        help="開始トークンのインデックス",
    )
    parser.add_argument(
        "--eos_idx",
        default=3,
        help="終了トークンのインデックス",
    )
    parser.add_argument(
        "--data_limit",
        type=int,
        help="使用データ数制限",
    )
    #
    # 学習設定関連
    #
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1024,
        help="バッチサイズ",
    )
    parser.add_argument(
        "--max_len",
        type=int,
        default=20,
        help="最大系列長",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="../save/",
        help="モデルの保存先",
    )
    #
    # モデルパラメータ関連
    #
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="エポック数",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="学習率",
    )
    parser.add_argument(
        "--optim_name", type=str, choices=["adam"], default="adam", help="オプティマイザー"
    )
    parser.add_argument(
        "--proj_share_weight",
        action="store_false",
        help="Embeddingの重みを共有するかどうか",
    )
    parser.add_argument(
        "--d_key",
        type=int,
        default=64,
        help="keyの重みの次元数",
    )
    parser.add_argument(
        "--d_value",
        type=int,
        default=64,
        help="valueの重みの次元数",
    )
    parser.add_argument(
        "--d_model",
        type=int,
        default=512,
        help="隠れ層の次元数, 単語埋め込み次元数",
    )
    parser.add_argument(
        "--d_inner_hid",
        type=int,
        default=2048,
        help="Feed-Forwardの隠れ層2層目の次元数",
    )
    parser.add_argument(
        "--n_layer",
        type=int,
        default=6,
        help="レイヤー数",
    )
    parser.add_argument(
        "--n_head",
        type=int,
        default=8,
        help="ヘッド数",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="ドロップアウト率",
    )
    parser.add_argument(
        "--use_pretrain",
        action="store_false",
        help="以前学習した際の重みを利用するかどうか",
    )
    #
    # 予測
    #
    parser.add_argument(
        "--predict_input",
        type=str,
        default="This is a pen.",
        help="予測入力文",
    )
    #
    # Optunaによるハイパラ調整の値
    #
    parser.add_argument("--n_trials", type=int, default=100, help="試行回数")
    #
    # その他
    #
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu"],
        default="cuda",
        help="使用デバイス",
    )
    parser.add_argument(
        "--use_visdom",
        action="store_false",
        help="visdomを使用するか",
    )
    parser.add_argument(
        "--use_amp",
        action="store_false",
        help="FP16学習をするかどうか",
    )

    args = parser.parse_args()
    args = refine_args(args)
    print("data limit:", args.data_limit)
    main(args)
