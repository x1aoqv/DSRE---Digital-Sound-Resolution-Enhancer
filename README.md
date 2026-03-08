# DSRE / Deep Sound Resolution Enhancer

## 简介 / Description

DSRE 是一款 **高性能音频增强工具**，可以将任何音频文件批量处理为 **高解析度（Hi-Res）音频**，无需大量计算资源即可快速处理大批量音频文件。

DSRE is a **high-performance audio enhancement tool** that can batch-convert any audio files into **high-resolution (Hi-Res) audio**.
Inspired by Sony DSEE HX, it uses a **non-deep-learning frequency enhancement algorithm**, allowing fast processing of large batches without heavy computation.

**主要特点 / Key Features:**

* **批量处理 / Batch Processing**：一次性转换多个音频文件 / Convert multiple audio files at once.
* **多格式支持 / Multiple Formats**：WAV、MP3、FLAC、M4A 等 / Supports WAV, MP3, FLAC, M4A, etc.
* **保持封面和元数据 / Preserves Cover & Metadata**：无需手动修改 / No manual editing required.
* **灵活参数控制 / Flexible Parameters**：调制次数、衰减幅度、高通滤波器等 / Modulation count, decay, high-pass filter, etc.
* **快速稳定 / Fast & Stable**：不依赖深度学习模型 / Does not rely on deep learning, fast processing.

---

## 安装与使用 / Installation & Usage

[下载 / Download](https://github.com/x1aoqv/DSRE---Digital-Sound-Resolution-Enhancer/releases/tag/v1.0.250908_beta)

---

## 参数说明 / Parameters

| 参数 / Parameter                               | 默认值 / Default | 说明 / Description                                                   |
| -------------------------------------------- | ------------- | ------------------------------------------------------------------ |
| 调制次数 (m) / Modulation count                  | 8             | 音频增强重复次数 / Number of enhancement repetitions, higher = more detail |
| 衰减幅度 (decay)                                 | 1.25          | 高频衰减控制 / High-frequency decay control                              |
| 预处理高通截止频率 / Pre-processing high-pass cutoff  | 3000 Hz       | 处理前高通滤波器 / Pre-enhancement high-pass filter                        |
| 后处理高通截止频率 / Post-processing high-pass cutoff | 16000 Hz      | 处理后高通滤波器 / Post-enhancement high-pass filter                       |
| 滤波器阶数 / Filter order                         | 11            | 高通滤波器阶数 / High-pass filter order                                   |
| 目标采样率 / Target sampling rate                 | 96000 Hz      | 输出音频采样率 / Output audio sample rate                                 |
| 输出格式 / Output format                         | ALAC / FLAC   | 选择 Hi-Res 输出格式 / Choose Hi-Res output format                       |


## 平台支持 / Platforms

| 平台 / Paltform                               | 支持状态 / Support | 说明 / Description                                                   |
| -------------------------------------------- | ------------- | ------------------------------------------------------------------ |
| MacOS                                        | 否 / No       | 目前无 / Not at present                                        |
| Unix                                         | 否 / No       | 目前无 / Not at present                                         |
| Windows                                      | 是 / Yes     | 原生支持 / native                                              |
| Linux                                        | 是 / Yes     | 使用Tkinter移植 / Transplant using tkinter                       |

## 使用方法 / Methods

* **安装Python3 / Install Python3**：python.org
* **创建虚拟环境 / Create Virtual Environment**：python3 -m venv ~/audio_env | source ~/audio_env/bin/activate
* **安装依赖 / Install Requirements**：pip install librosa resampy numpy soundfile scipy
* **运行 / Run It**：WAV、MP3、FLAC、M4A 等 / python3 dsre.py | deactivate


---
