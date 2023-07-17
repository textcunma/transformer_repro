import os
import torch
import argparse
import onnxruntime
import numpy as np

from transformer_repro.scripts.predict import PredictProcess
from transformer_repro.utils.utils import ids_to_sentence, trim_eos, print_time


class OnnxPredictProcess(PredictProcess):
    """ONNXによる予測処理"""

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__(args)
        self.d_model = args.d_model  # 隠れ次元数

    def __call__(self, predict_input: str = "What's up?") -> None:
        """

        Args:
            predict_input (str): 予測したい英語文章

        Notes:
            通常の推論と異なり、ONNXでは'tensor'ではなく'numpy'を利用するため、'predict.py'からいくつかの変更あり
        """

        # ONNXモデルを作成
        if not os.path.exists("../save/checkpoint.onnx"):
            self.convert_onnx()

        src_seq = self.cd.raw2id_numpy(
            self.data_mode, self.en_vocab, predict_input
        )

        zero_index = np.argmin(src_seq)
        src_pos = np.array(
            list(range(1, zero_index + 1)) + [0] * (self.max_len - zero_index)
        )

        src_seq = np.reshape(src_seq, (1, self.max_len)).astype(
            np.int64
        )  # (22,) -> (1, 22)
        src_pos = np.reshape(src_pos, (1, self.max_len)).astype(np.int64)

        result_seq = self._run_onnx(src_seq, src_pos)
        pred_ids = result_seq[0][0].data.tolist()
        predict_jp_sentence = ids_to_sentence(
            self.jp_vocab, trim_eos(pred_ids, self.eos_idx)
        )

        print(f"src:{predict_input}")
        print(f"out:{predict_jp_sentence}")

    @print_time
    def _run_onnx(self, src_seq, src_pos):
        """推論処理

        Notes:
            onnxruntime.get_device() -> 'CPU' or 'GPU'
            onnxruntime.get_available_providers() -> ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
            TensorRTを使用するためには, buildする必要があるそう
        """
        if self.device == 'cuda' and onnxruntime.get_device() == "GPU":
            print("GPU version")
            model_session = onnxruntime.InferenceSession(
                "../save/checkpoint.onnx", providers=["CUDAExecutionProvider"]
            )
        else:
            print("CPU version")
            model_session = onnxruntime.InferenceSession(
                "../save/checkpoint.onnx", providers=["CPUExecutionProvider"]
            )

        inputs = {"src_seq": src_seq, "src_pos": src_pos}
        predict_jp_sentence = model_session.run(["output"], inputs)
        return predict_jp_sentence

    def convert_onnx(self) -> None:
        """PyTorchモデルをONNXモデルに変換"""

        self.model.eval()

        dummy_src_seq = torch.unsqueeze(torch.randperm(self.max_len), dim=0).to(self.device)

        dummy_src_pos = torch.tensor(
            [i for i in range(1, self.max_len + 1)],
            dtype=torch.long,
        ).unsqueeze(dim=0).to(self.device)

        dummy_input = tuple([tuple([dummy_src_seq, dummy_src_pos])])

        input_names = tuple(["src_seq", "src_pos"])
        output_name = ["output"]

        torch.onnx.export(
            self.model,
            dummy_input,
            f"../save/checkpoint.onnx",
            export_params=True,
            opset_version=18,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_name,
        )
