# -*- coding: utf-8 -*-
"""
F0 (基频) 提取后端集合

支持的方法：
    - dio      : PyWORLD DIO（快速，由 tsubaki_processor.py 内置实现）
    - harvest  : PyWORLD Harvest（精确，由 tsubaki_processor.py 内置实现）
    - crepe    : 基于 torchcrepe 的神经网络音高估计（鲁棒，抗噪）
    - rmvpe    : 基于 RMVPE 深度模型的音高估计（对人声极为鲁棒，
                 是目前 SVC/SVS 流水线中最常用的高质量 F0 提取器）

本模块只负责"原始提取"：输入整段音频 (mono, float, 任意采样率)，
输出 (f0_hz, t_sec) —— f0 为 Hz，未发声帧 = 0.0；t 为对应时间戳（秒）。
后续的插值 / 平滑 / 写入 SVP・USTX 均由 tsubaki_processor.py 统一处理，
因此 CREPE / RMVPE 的输出可以与 DIO / Harvest 无缝衔接进入既有导出管线。

依赖（均为可选，缺失时仅影响对应方法，不影响 dio/harvest）：
    - torch
    - torchcrepe   (CREPE 神经网络，自带模型权重)
    - librosa      (RMVPE 的梅尔频谱 + 重采样；MFA 自身也依赖 librosa)

RMVPE 模型权重 (rmvpe.pt, 约 180MB) 不随本项目分发，需要用户自行下载：
    https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/rmvpe.pt
并放置到 <项目根>/models/rmvpe/rmvpe.pt，或通过环境变量 RMVPE_MODEL_PATH 指定路径。
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 可用性检测（全部懒加载导入，避免缺少 torch 时影响 dio/harvest 正常工作）
# ---------------------------------------------------------------------------

def torch_available() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


def torchcrepe_available() -> bool:
    try:
        import torchcrepe  # noqa: F401
        return True
    except ImportError:
        return False


def cuda_available() -> bool:
    """
    真正可用性检测：先查 is_available()，再做一次 smoke-test tensor 分配。
    某些 PyTorch 安装虽未编译 CUDA 支持，但 is_available() 因驱动存在仍会
    返回 True；实际使用时才在 _lazy_init 抛出 AssertionError。
    这里提前捕获，保证 _select_device() 不会错误地返回 "cuda"。
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return False
        # Smoke-test: 实际分配一个 CUDA tensor，触发 _lazy_init
        # 若 torch 未编译 CUDA 支持，这里会抛出 AssertionError
        torch.zeros(1, device="cuda")
        return True
    except Exception:
        return False


def rmvpe_model_path() -> Path:
    """RMVPE 权重文件路径：优先读取环境变量 RMVPE_MODEL_PATH，
    否则使用 <本文件所在目录>/models/rmvpe/rmvpe.pt"""
    env_path = os.environ.get("RMVPE_MODEL_PATH")
    if env_path:
        return Path(env_path)
    return Path(__file__).resolve().parent / "models" / "rmvpe" / "rmvpe.pt"


def rmvpe_available() -> bool:
    return torch_available() and rmvpe_model_path().exists()


def get_f0_backend_status() -> Dict[str, object]:
    """供 /api/pipeline/status 使用：报告各 F0 后端的可用性"""
    has_torch = torch_available()
    has_cuda = cuda_available() if has_torch else False
    has_crepe = has_torch and torchcrepe_available()
    rmvpe_path = rmvpe_model_path()

    return {
        "dio": {"available": True},
        "harvest": {"available": True},
        "crepe": {
            "available": has_crepe,
            "torch": has_torch,
            "torchcrepe": torchcrepe_available(),
            "cuda": has_cuda,
        },
        "rmvpe": {
            "available": rmvpe_available(),
            "torch": has_torch,
            "model_path": str(rmvpe_path),
            "model_found": rmvpe_path.exists(),
            "cuda": has_cuda,
        },
    }


# ---------------------------------------------------------------------------
# 通用工具
# ---------------------------------------------------------------------------

def _select_device(device: Optional[str] = "auto") -> str:
    device = (device or "auto").lower()
    if device not in ("auto", ""):
        return device
    return "cuda" if cuda_available() else "cpu"


def _resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """将单通道 float 音频从 sr_in 重采样到 sr_out。"""
    if sr_in == sr_out:
        return np.asarray(x, dtype=np.float32)

    try:
        import librosa
        return librosa.resample(
            np.asarray(x, dtype=np.float64), orig_sr=sr_in, target_sr=sr_out
        ).astype(np.float32)
    except ImportError:
        pass

    try:
        import resampy
        return resampy.resample(np.asarray(x, dtype=np.float32), sr_in, sr_out).astype(np.float32)
    except ImportError:
        pass

    # 兜底：numpy 线性插值重采样
    x = np.asarray(x, dtype=np.float64)
    duration = len(x) / float(sr_in)
    n_out = max(1, int(round(duration * sr_out)))
    x_old = np.linspace(0.0, duration, num=len(x), endpoint=False)
    x_new = np.linspace(0.0, duration, num=n_out, endpoint=False)
    return np.interp(x_new, x_old, x).astype(np.float32)


# ---------------------------------------------------------------------------
# CREPE (torchcrepe)
# ---------------------------------------------------------------------------

def _run_crepe_on_device(
    x: np.ndarray,
    sr: int,
    hop_length: int,
    fmin: float,
    fmax: float,
    model_size: str,
    batch_size: int,
    device: str,
    periodicity_threshold: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    在指定 device 上执行 torchcrepe 推理，返回 (f0, t)。
    调用方负责捕获 CUDA 相关异常并回退 CPU。
    """
    import torch
    import torchcrepe

    audio_tensor = torch.from_numpy(x).unsqueeze(0).to(device)

    pitch, periodicity = torchcrepe.predict(
        audio_tensor,
        sr,
        hop_length,
        fmin=fmin,
        fmax=fmax,
        model=model_size,
        batch_size=batch_size,
        device=device,
        return_periodicity=True,
        pad=True,
    )

    # 周期性中值滤波 + 音高均值滤波，抑制抖动
    periodicity = torchcrepe.filter.median(periodicity, 3)
    pitch = torchcrepe.filter.mean(pitch, 3)

    # 低置信度帧标记为未发声 (NaN → 之后转为 0)
    pitch = torchcrepe.threshold.At(periodicity_threshold)(pitch, periodicity)

    f0 = pitch.squeeze(0).detach().cpu().numpy().astype(np.float64)
    n_frames = f0.shape[0]
    t = np.arange(n_frames, dtype=np.float64) * (hop_length / float(sr))
    return f0, t


def extract_f0_crepe(
    audio: np.ndarray,
    sr: int,
    f0_floor: float = 71.0,
    f0_ceil: float = 800.0,
    frame_period_ms: float = 10.0,
    model_size: str = "full",
    device: str = "auto",
    batch_size: int = 512,
    periodicity_threshold: float = 0.21,
) -> Tuple[np.ndarray, np.ndarray]:
    """使用 torchcrepe (CREPE 神经网络) 提取 F0。

    Args:
        audio: 单通道音频，float，任意采样率。
        sr: 采样率。
        f0_floor / f0_ceil: 允许的频率范围 (Hz)。
        frame_period_ms: 帧间隔（毫秒），决定 hop_length。
        model_size: 'full' 或 'tiny'。
        device: 'auto' / 'cpu' / 'cuda'。
        batch_size: 推理批大小。
        periodicity_threshold: 周期性置信度阈值，低于该值的帧视为未发声 (f0=0)。

    Returns:
        (f0_hz, t_sec): 未发声帧 f0 = 0.0
    """
    import torchcrepe

    if not torchcrepe_available():
        raise ImportError("torchcrepe 未安装，请运行: pip install torchcrepe")

    device = _select_device(device)

    x = np.asarray(audio, dtype=np.float32)
    if x.ndim > 1:
        x = np.mean(x, axis=1)

    hop_length = max(1, int(round(sr * frame_period_ms / 1000.0)))
    fmin = max(float(f0_floor), 1.0)
    fmax = min(float(f0_ceil), float(torchcrepe.MAX_FMAX))

    try:
        f0, t = _run_crepe_on_device(
            x, sr, hop_length, fmin, fmax,
            model_size, batch_size, device, periodicity_threshold,
        )
    except (AssertionError, RuntimeError) as e:
        # Torch 未编译 CUDA 支持 / 显卡初始化失败 → 自动回退到 CPU
        if device != "cpu" and ("CUDA" in str(e) or "cuda" in str(e)):
            logger.warning(
                f"CREPE CUDA 初始化失败，自动回退到 CPU: {e}"
            )
            device = "cpu"
            f0, t = _run_crepe_on_device(
                x, sr, hop_length, fmin, fmax,
                model_size, batch_size, device, periodicity_threshold,
            )
        else:
            raise

    f0 = np.nan_to_num(f0, nan=0.0, posinf=0.0, neginf=0.0)
    f0[(f0 > 0) & ((f0 < f0_floor * 0.5) | (f0 > f0_ceil * 1.2))] = 0.0

    return f0, t


# ---------------------------------------------------------------------------
# RMVPE
#
# 模型结构精确复刻自 RVC-Project/Retrieval-based-Voice-Conversion-WebUI
# 的 infer/lib/rmvpe.py（去除了 DirectML / JIT / ONNX / IPEX 等与本项目
# 无关的分支），以保证可直接加载社区分发的 rmvpe.pt 权重文件。
# ---------------------------------------------------------------------------

_RMVPE_CACHE: Dict[Tuple[str, str], "RMVPEF0Extractor"] = {}


def _build_rmvpe_modules():
    """延迟构建 RMVPE 的 torch.nn 模块类（避免在未安装 torch 时报错）。"""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from librosa.filters import mel as librosa_mel_fn

    class BiGRU(nn.Module):
        def __init__(self, input_features, hidden_features, num_layers):
            super().__init__()
            self.gru = nn.GRU(
                input_features,
                hidden_features,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=True,
            )

        def forward(self, x):
            return self.gru(x)[0]

    class ConvBlockRes(nn.Module):
        def __init__(self, in_channels, out_channels, momentum=0.01):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3),
                          stride=(1, 1), padding=(1, 1), bias=False),
                nn.BatchNorm2d(out_channels, momentum=momentum),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3),
                          stride=(1, 1), padding=(1, 1), bias=False),
                nn.BatchNorm2d(out_channels, momentum=momentum),
                nn.ReLU(),
            )
            if in_channels != out_channels:
                self.shortcut = nn.Conv2d(in_channels, out_channels, (1, 1))

        def forward(self, x):
            if not hasattr(self, "shortcut"):
                return self.conv(x) + x
            return self.conv(x) + self.shortcut(x)

    class ResEncoderBlock(nn.Module):
        def __init__(self, in_channels, out_channels, kernel_size, n_blocks=1, momentum=0.01):
            super().__init__()
            self.n_blocks = n_blocks
            self.conv = nn.ModuleList()
            self.conv.append(ConvBlockRes(in_channels, out_channels, momentum))
            for _ in range(n_blocks - 1):
                self.conv.append(ConvBlockRes(out_channels, out_channels, momentum))
            self.kernel_size = kernel_size
            if self.kernel_size is not None:
                self.pool = nn.AvgPool2d(kernel_size=kernel_size)

        def forward(self, x):
            for conv in self.conv:
                x = conv(x)
            if self.kernel_size is not None:
                return x, self.pool(x)
            return x

    class Encoder(nn.Module):
        def __init__(self, in_channels, in_size, n_encoders, kernel_size, n_blocks,
                      out_channels=16, momentum=0.01):
            super().__init__()
            self.n_encoders = n_encoders
            self.bn = nn.BatchNorm2d(in_channels, momentum=momentum)
            self.layers = nn.ModuleList()
            self.latent_channels = []
            for _ in range(self.n_encoders):
                self.layers.append(
                    ResEncoderBlock(in_channels, out_channels, kernel_size, n_blocks, momentum=momentum)
                )
                self.latent_channels.append([out_channels, in_size])
                in_channels = out_channels
                out_channels *= 2
                in_size //= 2
            self.out_size = in_size
            self.out_channel = out_channels

        def forward(self, x):
            concat_tensors: List["torch.Tensor"] = []
            x = self.bn(x)
            for layer in self.layers:
                t, x = layer(x)
                concat_tensors.append(t)
            return x, concat_tensors

    class Intermediate(nn.Module):
        def __init__(self, in_channels, out_channels, n_inters, n_blocks, momentum=0.01):
            super().__init__()
            self.n_inters = n_inters
            self.layers = nn.ModuleList()
            self.layers.append(ResEncoderBlock(in_channels, out_channels, None, n_blocks, momentum))
            for _ in range(self.n_inters - 1):
                self.layers.append(ResEncoderBlock(out_channels, out_channels, None, n_blocks, momentum))

        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            return x

    class ResDecoderBlock(nn.Module):
        def __init__(self, in_channels, out_channels, stride, n_blocks=1, momentum=0.01):
            super().__init__()
            out_padding = (0, 1) if stride == (1, 2) else (1, 1)
            self.n_blocks = n_blocks
            self.conv1 = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=(3, 3),
                                    stride=stride, padding=(1, 1),
                                    output_padding=out_padding, bias=False),
                nn.BatchNorm2d(out_channels, momentum=momentum),
                nn.ReLU(),
            )
            self.conv2 = nn.ModuleList()
            self.conv2.append(ConvBlockRes(out_channels * 2, out_channels, momentum))
            for _ in range(n_blocks - 1):
                self.conv2.append(ConvBlockRes(out_channels, out_channels, momentum))

        def forward(self, x, concat_tensor):
            x = self.conv1(x)
            x = torch.cat((x, concat_tensor), dim=1)
            for conv2 in self.conv2:
                x = conv2(x)
            return x

    class Decoder(nn.Module):
        def __init__(self, in_channels, n_decoders, stride, n_blocks, momentum=0.01):
            super().__init__()
            self.layers = nn.ModuleList()
            self.n_decoders = n_decoders
            for _ in range(self.n_decoders):
                out_channels = in_channels // 2
                self.layers.append(ResDecoderBlock(in_channels, out_channels, stride, n_blocks, momentum))
                in_channels = out_channels

        def forward(self, x, concat_tensors):
            for i, layer in enumerate(self.layers):
                x = layer(x, concat_tensors[-1 - i])
            return x

    class DeepUnet(nn.Module):
        def __init__(self, kernel_size, n_blocks, en_de_layers=5, inter_layers=4,
                      in_channels=1, en_out_channels=16):
            super().__init__()
            self.encoder = Encoder(in_channels, 128, en_de_layers, kernel_size, n_blocks, en_out_channels)
            self.intermediate = Intermediate(
                self.encoder.out_channel // 2, self.encoder.out_channel, inter_layers, n_blocks
            )
            self.decoder = Decoder(self.encoder.out_channel, en_de_layers, kernel_size, n_blocks)

        def forward(self, x):
            x, concat_tensors = self.encoder(x)
            x = self.intermediate(x)
            x = self.decoder(x, concat_tensors)
            return x

    class E2E(nn.Module):
        def __init__(self, n_blocks, n_gru, kernel_size, en_de_layers=5, inter_layers=4,
                      in_channels=1, en_out_channels=16):
            super().__init__()
            self.unet = DeepUnet(kernel_size, n_blocks, en_de_layers, inter_layers, in_channels, en_out_channels)
            self.cnn = nn.Conv2d(en_out_channels, 3, (3, 3), padding=(1, 1))
            if n_gru:
                self.fc = nn.Sequential(
                    BiGRU(3 * 128, 256, n_gru),
                    nn.Linear(512, 360),
                    nn.Dropout(0.25),
                    nn.Sigmoid(),
                )
            else:
                self.fc = nn.Sequential(
                    nn.Linear(3 * 128, 360), nn.Dropout(0.25), nn.Sigmoid()
                )

        def forward(self, mel):
            mel = mel.transpose(-1, -2).unsqueeze(1)
            x = self.cnn(self.unet(mel)).transpose(1, 2).flatten(-2)
            x = self.fc(x)
            return x

    class MelSpectrogram(nn.Module):
        def __init__(self, is_half, n_mel_channels, sampling_rate, win_length, hop_length,
                      n_fft=None, mel_fmin=0, mel_fmax=None, clamp=1e-5):
            super().__init__()
            n_fft = win_length if n_fft is None else n_fft
            self.hann_window: Dict[str, "torch.Tensor"] = {}
            mel_basis = librosa_mel_fn(
                sr=sampling_rate, n_fft=n_fft, n_mels=n_mel_channels,
                fmin=mel_fmin, fmax=mel_fmax, htk=True,
            )
            mel_basis = torch.from_numpy(mel_basis).float()
            self.register_buffer("mel_basis", mel_basis)
            self.n_fft = n_fft
            self.hop_length = hop_length
            self.win_length = win_length
            self.sampling_rate = sampling_rate
            self.n_mel_channels = n_mel_channels
            self.clamp = clamp
            self.is_half = is_half

        def forward(self, audio, center=True):
            keyshift_key = "0_" + str(audio.device)
            if keyshift_key not in self.hann_window:
                self.hann_window[keyshift_key] = torch.hann_window(self.win_length).to(audio.device)

            fft = torch.stft(
                audio,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                window=self.hann_window[keyshift_key],
                center=center,
                return_complex=True,
            )
            magnitude = torch.sqrt(fft.real.pow(2) + fft.imag.pow(2))
            mel_output = torch.matmul(self.mel_basis, magnitude)
            if self.is_half:
                mel_output = mel_output.half()
            return torch.log(torch.clamp(mel_output, min=self.clamp))

    return {
        "E2E": E2E,
        "MelSpectrogram": MelSpectrogram,
    }


class RMVPEF0Extractor:
    """RMVPE F0 提取器（懒加载模型权重，输出 10ms 帧间隔的 16kHz 域 F0）。"""

    HOP_LENGTH = 160  # @ 16kHz => 10ms / 帧

    def __init__(self, model_path: str, device: str = "cpu"):
        import torch

        self.device = device
        self.is_half = False

        modules = _build_rmvpe_modules()
        E2E = modules["E2E"]
        MelSpectrogram = modules["MelSpectrogram"]

        self.mel_extractor = MelSpectrogram(
            self.is_half, 128, 16000, 1024, self.HOP_LENGTH, None, 30, 8000
        ).to(device)

        model = E2E(4, 1, (2, 2))
        ckpt = torch.load(model_path, map_location="cpu")
        if isinstance(ckpt, dict) and "model" in ckpt and not any(
            k.startswith("unet") or k.startswith("cnn") or k.startswith("fc") for k in ckpt
        ):
            ckpt = ckpt["model"]
        model.load_state_dict(ckpt)
        model.eval()
        model = model.float().to(device)
        self.model = model

        cents_mapping = 20 * np.arange(360) + 1997.3794084376191
        self.cents_mapping = np.pad(cents_mapping, (4, 4))  # 368

    def _mel2hidden(self, mel):
        import torch
        import torch.nn.functional as F

        with torch.no_grad():
            n_frames = mel.shape[-1]
            n_pad = 32 * ((n_frames - 1) // 32 + 1) - n_frames
            if n_pad > 0:
                mel = F.pad(mel, (0, n_pad), mode="constant")
            hidden = self.model(mel.float())
            return hidden[:, :n_frames]

    def _to_local_average_cents(self, salience: np.ndarray, thred: float = 0.03) -> np.ndarray:
        center = np.argmax(salience, axis=1)
        salience = np.pad(salience, ((0, 0), (4, 4)))
        center += 4
        todo_salience = []
        todo_cents_mapping = []
        starts = center - 4
        ends = center + 5
        for idx in range(salience.shape[0]):
            todo_salience.append(salience[idx, starts[idx]:ends[idx]])
            todo_cents_mapping.append(self.cents_mapping[starts[idx]:ends[idx]])
        todo_salience = np.array(todo_salience)
        todo_cents_mapping = np.array(todo_cents_mapping)
        product_sum = np.sum(todo_salience * todo_cents_mapping, axis=1)
        weight_sum = np.sum(todo_salience, axis=1)
        weight_sum = np.where(weight_sum == 0, 1e-9, weight_sum)
        devided = product_sum / weight_sum
        maxx = np.max(salience, axis=1)
        devided[maxx <= thred] = 0
        return devided

    def infer_from_audio(self, audio_16k: np.ndarray, thred: float = 0.03) -> np.ndarray:
        import torch

        audio_tensor = torch.from_numpy(np.asarray(audio_16k, dtype=np.float32))
        mel = self.mel_extractor(audio_tensor.to(self.device).unsqueeze(0), center=True)
        hidden = self._mel2hidden(mel)
        hidden = hidden.squeeze(0).detach().cpu().numpy()

        cents_pred = self._to_local_average_cents(hidden, thred=thred)
        f0 = 10 * (2 ** (cents_pred / 1200))
        f0[f0 == 10] = 0.0
        return f0


def _get_rmvpe_extractor(model_path: str, device: str) -> RMVPEF0Extractor:
    key = (model_path, device)
    extractor = _RMVPE_CACHE.get(key)
    if extractor is None:
        logger.info(f"加载 RMVPE 模型: {model_path} (device={device})")
        try:
            extractor = RMVPEF0Extractor(model_path, device=device)
        except (AssertionError, RuntimeError) as e:
            # Torch 未编译 CUDA 支持 / 显卡初始化失败 → 自动回退到 CPU
            if device != "cpu" and ("CUDA" in str(e) or "cuda" in str(e)):
                logger.warning(
                    f"RMVPE CUDA 初始化失败，自动回退到 CPU: {e}"
                )
                extractor = RMVPEF0Extractor(model_path, device="cpu")
                # 同时缓存 cpu key，避免下次再次尝试 CUDA
                _RMVPE_CACHE[(model_path, "cpu")] = extractor
            else:
                raise
        _RMVPE_CACHE[key] = extractor
    return extractor


def extract_f0_rmvpe(
    audio: np.ndarray,
    sr: int,
    f0_floor: float = 71.0,
    f0_ceil: float = 800.0,
    device: str = "auto",
    model_path: Optional[str] = None,
    thred: float = 0.03,
) -> Tuple[np.ndarray, np.ndarray]:
    """使用 RMVPE 深度模型提取 F0。

    RMVPE 内部固定工作在 16kHz / hop=160 (10ms 一帧)，任意输入采样率会
    自动重采样到 16kHz。

    Returns:
        (f0_hz, t_sec): 未发声帧 f0 = 0.0
    """
    if not torch_available():
        raise ImportError("torch 未安装，请运行: pip install torch")

    mp = Path(model_path) if model_path else rmvpe_model_path()
    if not mp.exists():
        raise FileNotFoundError(
            f"未找到 RMVPE 模型权重: {mp}\n"
            f"请从 https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/rmvpe.pt "
            f"下载 rmvpe.pt，放置到该路径，或设置环境变量 RMVPE_MODEL_PATH 指向权重文件。"
        )

    device = _select_device(device)

    x = np.asarray(audio, dtype=np.float32)
    if x.ndim > 1:
        x = np.mean(x, axis=1)

    target_sr = 16000
    x16 = _resample(x, sr, target_sr) if sr != target_sr else x

    extractor = _get_rmvpe_extractor(str(mp), device)
    f0 = extractor.infer_from_audio(x16, thred=thred)

    f0 = np.asarray(f0, dtype=np.float64)
    f0 = np.nan_to_num(f0, nan=0.0, posinf=0.0, neginf=0.0)
    f0[(f0 > 0) & ((f0 < f0_floor * 0.5) | (f0 > f0_ceil * 1.2))] = 0.0

    n_frames = f0.shape[0]
    t = np.arange(n_frames, dtype=np.float64) * (RMVPEF0Extractor.HOP_LENGTH / float(target_sr))

    return f0, t
