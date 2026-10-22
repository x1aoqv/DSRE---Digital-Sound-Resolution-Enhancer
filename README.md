# DSRE — Digital Sound Resolution Enhancer

> **Superseded / 已停更**  
> This project is **not** Sony DSEE HX. It was an empirical SSB-style approximation and is **no longer maintained**.  
> 本项目**不是** Sony DSEE HX，而是多次 SSB 一类启发式近似，**已停止维护**。  
>  
> For a rebuild of the real Music Center DSEE HX filter (bit-level match on checked samples):  
> 真正的 Music Center DSEE HX 滤镜重建（已对照样例上可位级一致）：  
> **→ [Sony DSEE HX — Reverse Engineering & Reimplementation](https://github.com/x1aoqv/dsee_hx_re)**  
>  
> Looking for the old GUI / how to run DSRE? See [issue #1](https://github.com/x1aoqv/DSRE---Digital-Sound-Resolution-Enhancer/issues/1) — that UI belongs to **this** repo only, not the new one.  
> 还在找旧界面 / 怎么跑 DSRE？见 [issue #1](https://github.com/x1aoqv/DSRE---Digital-Sound-Resolution-Enhancer/issues/1)——那是**本仓库**的界面，不是新库。

---

## Why this exists (history) / 为什么还有这个仓库

DSRE tried to restore “Hi-Res-like” highs with repeated **SSB (single-sideband)**-style modulation and filter sweeps. It only *looked* a bit like stacking energy upward; that is **not** how DSEE HX works.

DSRE 曾用多次 **SSB（单边带）** 调制与扫参去近似「Hi-Res 高频」。频谱上看起来有点像往上叠能量，但那是**经验主义错误**，**不是** DSEE HX 的原理。

**Do not treat DSRE as a description of DSEE HX.**  
**请不要把 DSRE 当成 DSEE HX 的原理说明。**

This repo stays public for history / downloads. Prefer the new repo above for anything related to real DSEE HX.

本仓库仅作历史存档与旧版下载；涉及真 DSEE HX 请用上面的新库。

---

## Legacy docs (unchanged below) / 以下为旧版说明（归档）

# DSRE / Deep Sound Resolution Enhancer

## Description / 简介

DSRE is a **high-performance audio enhancement tool** that can batch-convert audio into **Hi-Res**-rate files using a **non-deep-learning** frequency enhancement algorithm (SSB-style heuristics). Fast batch processing; **not** a reverse of Sony DSEE HX.

DSRE 是一款 **高性能音频增强工具**，用**非深度学习**的频域增强（SSB 类启发式）批量升到高采样率。处理快；**不是** Sony DSEE HX 的逆向。

**Key features / 主要特点:**

* **Batch processing / 批量处理**
* **Multiple formats / 多格式**：WAV、MP3、FLAC、M4A, etc.
* **Preserves cover & metadata / 保持封面与元数据**
* **Tunable parameters / 可调参数**：modulation count, decay, high-pass, etc.
* **Fast / 快速**：no neural net

---

## Install / 安装与使用

[Download / 下载](https://github.com/x1aoqv/DSRE---Digital-Sound-Resolution-Enhancer/releases/tag/v1.0.250908_beta)

---

## Parameters / 参数说明

| Parameter / 参数 | Default / 默认 | Description / 说明 |
|--|--|--|
| Modulation count (m) / 调制次数 | 8 | Enhancement repetitions / 重复次数 |
| Decay / 衰减幅度 | 1.25 | High-frequency decay / 高频衰减 |
| Pre high-pass / 预处理高通 | 3000 Hz | Before enhancement / 增强前 |
| Post high-pass / 后处理高通 | 16000 Hz | After enhancement / 增强后 |
| Filter order / 滤波器阶数 | 11 | High-pass order / 高通阶数 |
| Target rate / 目标采样率 | 96000 Hz | Output sample rate / 输出采样率 |
| Output format / 输出格式 | ALAC / FLAC | Hi-Res container / 封装格式 |

---
