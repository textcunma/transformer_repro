import argparse
import torch.optim as optim
import optuna

from dataclasses import dataclass
from transformer_repro.utils.utils import calc_bleu
from transformer_repro.scripts.train_test import TrainTestProcess


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


class OptunaProcess(TrainTestProcess):
    """Optunaライブラリを用いたハイパラ最適化

    オプティマイザ, 活性化関数, 損失関数, その他パラメーター等の調整
    """

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__(args)
        self.args = args
        self.n_trials: int = args.n_trials  # 試行回数

    def __call__(self, *args, **kwargs):
        study = optuna.create_study(direction="maximize")  # 目的関数を最大化
        study.optimize(self.objective, n_trials=self.n_trials)

        # 最適化の結果を表示
        trial = study.best_trial
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))

    @staticmethod
    def get_optimizer(trial, model):
        """最適化するオプティマイザの選択範囲を決定

        オプティマイザの種類, 学習率をそれぞれどのような範囲でとるかを設定

        Args:
            trial ():
            model ():

        Returns:
            optimizer ():
        """
        candidate_opt_names = ["Adam", "RMSprop", "SGD", "MomentumSGD"]
        opt_names = trial.suggest_categorical("optimizer", candidate_opt_names)
        weight_decay = trial.suggest_float("weight_decay", 1e-10, 1e-3)

        if opt_names == candidate_opt_names[0]:
            # Adamの場合
            adam_lr = trial.suggest_float("adam_lr", 1e-5, 1e-1)
            optimizer = optim.Adam(
                model.parameters(), lr=adam_lr, weight_decay=weight_decay
            )

        elif opt_names == candidate_opt_names[1]:
            # RMSpropの場合
            rmsprop_lr = trial.suggest_float("rmsprop_lr", 1e-5, 1e-1)
            optimizer = optim.RMSprop(
                model.parameters(), lr=rmsprop_lr, weight_decay=weight_decay
            )

        elif opt_names == candidate_opt_names[2]:
            # SGDの場合
            sgd_lr = trial.suggest_float("sgd_lr", 1e-3, 1e-1)
            optimizer = optim.SGD(
                model.parameters(), lr=sgd_lr, weight_decay=weight_decay
            )

        else:
            # MomentumSGDの場合
            momentum_sgd_lr = trial.suggest_float("momentum_sgd_lr", 1e-5, 1e-1)
            optimizer = optim.SGD(
                model.parameters(),
                lr=momentum_sgd_lr,
                momentum=0.9,
                weight_decay=weight_decay,
            )
        return optimizer

    @staticmethod
    def get_n_layers(trial):
        """レイヤー数の選択"""
        n_layers = trial.suggest_int("num_layers", 1, 3)
        return n_layers

    @staticmethod
    def get_dropout(trial):
        """ドロップアウト率の選択"""
        dropout = trial.suggest_float("dropout_rate", 0.0, 0.5)
        return dropout

    def objective(self, trial):
        """目的関数

        BLEUスコアを最適化(最大化)する
        """
        # 試行の際の値
        dropout = self.get_dropout(trial)
        n_layer = self.get_n_layers(trial)

        # 更新
        self.args.dropout = dropout
        self.args.n_layer = n_layer

        # 更新した値を用いてモデルを初期化
        self.model = self._init_model(self.args, self.vocab_size)

        # オプティマイザの設定
        self.optimizer = self.get_optimizer(trial, self.model)

        best_valid_bleu = 0.0

        for epoch in range(self.epochs):
            lr_info = LearnInfo(0.0, [], [], 0.0, 0.0, [], [], 0.0)
            self._train(lr_info, epoch)

            # 検証工程
            self._valid(lr_info)

            self.scheduler.step(lr_info.valid_loss)

            lr_info.train_loss /= float(self.train_dataloader.sampler.num_samples)  # type: ignore
            lr_info.valid_loss /= float(self.valid_dataloader.sampler.num_samples)

            lr_info.valid_bleu = calc_bleu(
                lr_info.valid_refs, lr_info.valid_hyps, self.eos_idx
            )

            if lr_info.valid_bleu > best_valid_bleu:
                best_valid_bleu = lr_info.valid_bleu

        print("best_valid_bleu: ", best_valid_bleu)
        return best_valid_bleu
