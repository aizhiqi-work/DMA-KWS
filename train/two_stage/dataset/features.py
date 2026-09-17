import os
import random
import torch
import torchaudio
from torch.utils.data import Dataset

import sys
sys.path.append("/nvme01/openkws/qbyt")
from models.processor import compute_fbank


class FeatureExtractor:
    def __init__(self, augment=True, wav_dir="/nvme01/openkws/libriphrase/segments"):
        """
        初始化 FeatureExtractor 类，进行数据增强设置及噪声文件加载。

        Args:
            augment (bool): 是否启用数据增强
            wav_dir (str): 音频文件存放目录
        """
        self.data_aug = {
            "speed_perturb": augment,
            "add_noise": augment,
            "noise_lists": "/nvme01/openkws/qbyt/dataset/noise.list"
        }
        self.wav_dir = wav_dir
        self.noise_lists = self._load_noise_files(self.data_aug["noise_lists"])

    def _load_noise_files(self, noise_file_path):
        """
        加载噪声文件列表。

        Args:
            noise_file_path (str): 噪声文件路径
        
        Returns:
            list: 噪声文件路径列表
        """
        if not os.path.exists(noise_file_path):
            raise FileNotFoundError(f"Noise list file not found: {noise_file_path}")
        
        with open(noise_file_path, "r") as file:
            return file.readlines()

    def _apply_speed_perturbation(self, waveform, sample_rate, speeds=[0.9, 1.0, 1.1]):
        """
        对音频进行速度扰动。

        Args:
            waveform (Tensor): 音频波形
            sample_rate (int): 采样率
            speeds (list): 可选速度列表

        Returns:
            Tensor: 加速或减速后的波形
        """
        speed = random.choice(speeds)
        if speed != 1.0:
            waveform, _ = torchaudio.sox_effects.apply_effects_tensor(
                waveform, sample_rate,
                [['speed', str(speed)], ['rate', str(sample_rate)]]
            )
        return waveform

    def _add_noise_with_snr(self, waveform, sample_rate, noise_lists, target_snr_db):
        """
        按照目标信噪比（SNR）将噪声加到音频中。

        Args:
            waveform (Tensor): 原始音频波形
            sample_rate (int): 采样率
            noise_lists (list): 噪声文件路径列表
            target_snr_db (float): 目标信噪比（dB）

        Returns:
            Tensor: 加噪声后的音频波形
        """
        # 随机选择噪声文件
        noise_path = random.choice(noise_lists).strip()
        noise_wav, noise_sr = torchaudio.load(noise_path)
        
        # 如果噪声采样率与原音频不一致，进行重采样
        if noise_sr != sample_rate:
            noise_wav = torchaudio.transforms.Resample(noise_sr, sample_rate)(noise_wav)

        # 调整噪声长度与原音频一致
        if noise_wav.shape[1] < waveform.shape[1]:
            repeat_times = (waveform.shape[1] // noise_wav.shape[1]) + 1
            noise_wav = noise_wav.repeat(1, repeat_times)[:, :waveform.shape[1]]
        else:
            noise_wav = noise_wav[:, :waveform.shape[1]]

        # 计算信号功率和噪声功率
        signal_power = torch.mean(waveform ** 2)
        noise_power = torch.mean(noise_wav ** 2)

        # 计算并调整噪声幅度以匹配目标SNR
        snr_linear = 10 ** (target_snr_db / 10)
        scaling_factor = torch.sqrt(signal_power / (snr_linear * noise_power))
        noise_wav *= scaling_factor

        # 将噪声加到原始音频上
        return waveform + noise_wav

    def process(self, wav_path):
        """
        处理音频文件，进行特征提取并进行数据增强。

        Args:
            wav_path (str): 音频文件路径

        Returns:
            dict: 特征处理后的样本，包括音频、特征和标签
        """
        wav_path = os.path.join(self.wav_dir, wav_path)
        waveform, sr = torchaudio.load(wav_path)

        # 数据增强：速度扰动
        if self.data_aug["speed_perturb"]:
            waveform = self._apply_speed_perturbation(waveform, sr)
        
        # 数据增强：加噪声
        if self.data_aug["add_noise"]:
            if random.random() < 0.5:
                snr = random.choice(range(5, 20))  # 随机选择信噪比范围
                waveform = self._add_noise_with_snr(waveform, sr, self.noise_lists, snr)
            else:
                # 保持原音频不变
                waveform = waveform  

        # 特征提取：计算 Mel频率倒谱系数 (MFCC)
        sample = compute_fbank({
            "key": wav_path,
            "wav": waveform,
            "sample_rate": sr
        }, num_mel_bins=80, frame_shift=10, frame_length=25, dither=0.1)
        return sample
