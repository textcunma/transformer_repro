import time
import visdom
import socket
import subprocess
import numpy as np

from contextlib import closing
from typing import List


class ShowVisdom:
    """Visdomで表示するためのクラス"""

    train_graph_label: dict = dict(
        title="Training Loss", xlabel="Training Iteration", ylabel="Loss"
    )
    valid_graph_label: dict = dict(
        title="Loss", xlabel="epoch", ylabel="Loss", legend=["train", "valid"]
    )
    bleu_graph_label: dict = dict(
        title="bleu", xlabel="epoch", ylabel="score", legend=["train", "valid"]
    )

    def __init__(self) -> None:
        self.vis = None
        self.proc = None
        self.port = 8097  # default port

        # portを開ける, visdomを起動
        self.__enter__()

        self.vis_window_train = {"iter": None, "loss": None}
        self.vis_window_valid = {"epoch": None, "loss": None}
        self.vis_window_bleu = {"epoch": None, "score": None}

        self.train_loss_list: List[float] = []  # 損失値リスト
        self.all_training_loss: List[float] = []  # 損失値総和リスト
        self.all_valid_loss: List[float] = []  # 検証値リスト
        self.all_training_bleu: List[float] = []  # 学習bleuリスト
        self.all_valid_bleu: List[float] = []  # 検証bleuリスト

    def __enter__(self) -> None:
        """リソースを確保

        Note:
            参考: https://takemikami.com/2021/07/17/PythonRedis-server.html
        """
        # 空きポートの探索
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
            sock.bind(("", 0))
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.port = sock.getsockname()[1]

        self.proc = subprocess.Popen(  # type: ignore
            ["python", "-m", "visdom.server", "-port", str(self.port)]
        )
        time.sleep(5)
        if self.proc.poll() is not None:  # type: ignore
            raise Exception("fail to start server")

        self.vis = visdom.Visdom(port=self.port)  # type: ignore
        if not self.vis.check_connection():  # type: ignore
            raise Exception("fail to connect visdom")

    def __exit__(self) -> None:
        """リソースを解放"""
        self.vis.close()  # type: ignore
        self.proc.kill()  # type: ignore

    def train_itr(
        self, loss: float, epoch: int, n_batches: int, train_iter: int
    ) -> None:
        """学習時のイテレーションごとの表示

        Args:
            loss (float): 損失値
            epoch (int): エポック数
            n_batches (int): ミニバッチ数
            train_iter (int): イテレーション数
        """
        self.train_loss_list.append(loss)

        x = np.arange(
            epoch * n_batches + train_iter, epoch * n_batches + train_iter + 1
        )

        if self.vis_window_train["iter"] is None:
            self.vis_window_train["iter"] = self.vis.line(  # type: ignore
                X=x,
                Y=np.asarray(self.train_loss_list),
                opts=self.train_graph_label,
            )
        else:
            self.vis.line(
                X=x,
                Y=np.asarray([np.mean(self.train_loss_list)]),
                win=self.vis_window_train["iter"],
                opts=self.train_graph_label,
                update="append",
            )

    def train_valid(self, train_loss: float, valid_loss: float) -> None:
        """エポックごとの損失値

        Args:
            train_loss (float): 学習損失
            valid_loss (float): 検証損失
        """
        self.all_training_loss.append(train_loss)
        self.all_valid_loss.append(valid_loss)

        if self.vis_window_valid["epoch"] is None:
            self.vis_window_valid["epoch"] = self.vis.line(  # type: ignore
                X=np.tile(np.arange(len(self.all_valid_loss)), (2, 1)).T,
                Y=np.column_stack(
                    (
                        np.asarray(self.all_training_loss),
                        np.asarray(self.all_valid_loss),
                    )
                ),
                opts=self.valid_graph_label,
            )
        else:
            self.vis.line(
                X=np.tile(np.arange(len(self.all_valid_loss)), (2, 1)).T,
                Y=np.column_stack(
                    (
                        np.asarray(self.all_training_loss),
                        np.asarray(self.all_valid_loss),
                    )
                ),
                win=self.vis_window_valid["epoch"],
                opts=self.valid_graph_label,
            )

    def bleu(self, train_bleu: float, valid_bleu: float) -> None:
        """エポックごとのBLEUスコア

        Args:
            train_bleu (float): 学習データにおけるBLEUスコア
            valid_bleu (float): 検証データにおけるBLEUスコア
        """
        self.all_training_bleu.append(train_bleu)
        self.all_valid_bleu.append(valid_bleu)

        if self.vis_window_bleu["epoch"] is None:
            self.vis_window_bleu["epoch"] = self.vis.line(  # type: ignore
                X=np.tile(np.arange(len(self.all_valid_bleu)), (2, 1)).T,
                Y=np.column_stack(
                    (
                        np.asarray(self.all_training_bleu),
                        np.asarray(self.all_valid_bleu),
                    )
                ),
                opts=self.bleu_graph_label,
            )
        else:
            self.vis.line(
                X=np.tile(np.arange(len(self.all_valid_bleu)), (2, 1)).T,
                Y=np.column_stack(
                    (
                        np.asarray(self.all_training_bleu),
                        np.asarray(self.all_valid_bleu),
                    )
                ),
                win=self.vis_window_bleu["epoch"],
                opts=self.bleu_graph_label,
            )
