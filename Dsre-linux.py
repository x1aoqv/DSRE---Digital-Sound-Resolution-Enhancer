import os
import sys
import traceback
from typing import Optional
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import queue

import subprocess
import tempfile
import librosa
import resampy

import numpy as np
import soundfile as sf
from scipy import signal

# ==================== Linux适配点1：FFmpeg路径处理 ====================
def add_ffmpeg_to_path():
    if hasattr(sys, "_MEIPASS"):  # 打包后的临时目录(Linux下pyinstaller同样适用)
        ffmpeg_dir = os.path.join(sys._MEIPASS, "ffmpeg")
    else:
        ffmpeg_dir = os.path.join(os.path.dirname(__file__), "ffmpeg")
    # Linux下PATH拼接格式与Windows一致，无需修改
    os.environ["PATH"] += os.pathsep + ffmpeg_dir

add_ffmpeg_to_path()

# ==================== 音频保存核心函数 ====================
def save_wav24_out(in_path, y_out, sr, out_path, fmt="ALAC", normalize=True):
    # 确保 shape 为 (n, ch)
    if y_out.ndim == 1:
        data = y_out[:, None]
    else:
        data = y_out.T if y_out.shape[0] < y_out.shape[1] else y_out

    # 转为 float32 并归一化
    data = data.astype(np.float32, copy=False)
    if normalize:
        peak = float(np.max(np.abs(data)))
        if peak > 1.0:
            data /= peak
    else:
        data = np.clip(data, -1.0, 1.0)

    # ==================== Linux适配点2：临时文件处理 ====================
    # Linux下tempfile默认在/tmp，需确保ffmpeg有权限访问
    tmp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tmp_wav.close()
    sf.write(tmp_wav.name, data, sr, subtype="FLOAT")

    fmt = fmt.upper()
    out_path = os.path.splitext(out_path)[0] + (".m4a" if fmt == "ALAC" else ".flac")

    codec_map = {"ALAC": "alac", "FLAC": "flac"}
    sample_fmt_map = {"ALAC": "s32p", "FLAC": "s32"}  # 强制 24bit 整数

    if fmt == "ALAC":
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            tmp_wav.name,
            "-i",
            in_path,
            "-map",
            "0:a",  # 临时 WAV 音频
            "-map",
            "1:v?",  # 封面
            "-map_metadata",
            "1",  # 元数据
            "-c:a",
            codec_map[fmt],
            "-sample_fmt",
            sample_fmt_map[fmt],
            "-c:v",
            "copy",
            out_path,
        ]
    elif fmt == "FLAC":
        # 提取封面图片
        cover_tmp = None
        try:
            cover_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
            cover_tmp.close()
            subprocess.run(
                ["ffmpeg", "-y", "-i", in_path, "-an", "-c:v", "copy", cover_tmp.name],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception:
            cover_tmp = None

        if cover_tmp and os.path.exists(cover_tmp.name):
            cmd = [
                "ffmpeg",
                "-y",
                "-i",
                tmp_wav.name,  # WAV 音频
                "-i",
                in_path,  # 元数据来源
                "-i",
                cover_tmp.name,  # 封面
                "-map",
                "0:a",  # 音频
                "-map",
                "2:v",  # 封面
                "-disposition:v",
                "attached_pic",
                "-map_metadata",
                "1",  # 元数据
                "-c:a",
                codec_map[fmt],
                "-sample_fmt",
                sample_fmt_map[fmt],
                "-c:v",
                "copy",
                out_path,
            ]
        else:
            cmd = [
                "ffmpeg",
                "-y",
                "-i",
                tmp_wav.name,
                "-i",
                in_path,
                "-map",
                "0:a",
                "-map_metadata",
                "1",
                "-c:a",
                codec_map[fmt],
                "-sample_fmt",
                sample_fmt_map[fmt],
                out_path,
            ]

    # ==================== Linux适配点3：subprocess调用兼容 ====================
    # Linux下无需修改ffmpeg命令，确保系统已安装ffmpeg
    subprocess.run(cmd, check=True)
    os.remove(tmp_wav.name)
    if fmt == "FLAC" and cover_tmp and os.path.exists(cover_tmp.name):
        os.remove(cover_tmp.name)

    return out_path

# ==================== DSP：SSB 单边带频移 ====================
def freq_shift_mono(x: np.ndarray, f_shift: float, d_sr: float) -> np.ndarray:
    N_orig = len(x)
    # pad 到 2 的幂次，便于 FFT/Hilbert 的实现效率
    N_padded = 1 << int(np.ceil(np.log2(max(1, N_orig))))
    S_hilbert = signal.hilbert(
        np.hstack((x, np.zeros(N_padded - N_orig, dtype=x.dtype)))
    )
    S_factor = np.exp(2j * np.pi * f_shift * d_sr * np.arange(0, N_padded))
    return (S_hilbert * S_factor)[:N_orig].real

def freq_shift_multi(x: np.ndarray, f_shift: float, d_sr: float) -> np.ndarray:
    return np.asarray([freq_shift_mono(x[i], f_shift, d_sr) for i in range(len(x))])

def zansei_impl(
    x: np.ndarray,
    sr: int,
    m: int = 8,
    decay: float = 1.25,
    pre_hp: float = 3000.0,
    post_hp: float = 16000.0,
    filter_order: int = 11,
    progress_cb=None,
    abort_cb=None,  # 新增回调
) -> np.ndarray:
    # 预处理高通
    b, a = signal.butter(filter_order, pre_hp / (sr / 2), "highpass")
    d_src = signal.filtfilt(b, a, x)

    d_sr = 1.0 / sr
    f_dn = freq_shift_mono if (x.ndim == 1) else freq_shift_multi
    d_res = np.zeros_like(x)

    for i in range(m):
        if abort_cb and abort_cb():
            break  # 立即退出处理
        shift_hz = sr * (i + 1) / (m * 2.0)
        d_res += f_dn(d_src, shift_hz, d_sr) * np.exp(-(i + 1) * decay)
        if progress_cb:
            progress_cb(i + 1, m)

    # 后处理高通
    b, a = signal.butter(filter_order, post_hp / (sr / 2), "highpass")
    d_res = signal.filtfilt(b, a, d_res)

    adp_power = float(np.mean(np.abs(d_res)))
    src_power = float(np.mean(np.abs(x)))
    adj_factor = src_power / (adp_power + src_power + 1e-12)

    y = (x + d_res) * adj_factor
    return y

# ==================== 后台工作类（替换Qt线程） ====================
class DSREWorker:
    def __init__(self, files, output_dir, params, callback_queue):
        self.files = files
        self.output_dir = output_dir
        self.params = params
        self.callback_queue = callback_queue
        self._abort = False
        self.thread = threading.Thread(target=self.run)

    def abort(self):
        self._abort = True

    def start(self):
        self.thread.start()

    def is_alive(self):
        return self.thread.is_alive()

    def _send_callback(self, callback_type, *args):
        """发送回调消息到主线程"""
        self.callback_queue.put((callback_type, args))

    def run(self):
        total = len(self.files)
        done = 0
        self._send_callback("overall_progress", done, total)

        for idx, in_path in enumerate(self.files, start=1):
            if self._abort:
                break

            fname = os.path.basename(in_path)
            self._send_callback("file_progress", idx, total, fname)
            self._send_callback("step_progress", 0, fname)

            try:
                # 读取
                self._send_callback("log", f"正在加载：{in_path}")
                y, sr = librosa.load(in_path, mono=False, sr=None)

                # 对齐为 (ch, n)
                if y.ndim == 1:
                    y = y[np.newaxis, :]
                # 重采样
                target_sr = int(self.params["target_sr"])
                if sr != target_sr:
                    self._send_callback("log", f"正在进行：{fname}: {sr} -> {target_sr}")
                    y = resampy.resample(y, sr, target_sr, filter="kaiser_fast")
                    sr = target_sr

                # 处理
                def step_cb(cur, m):
                    pct = int(cur * 100 / max(1, m))
                    self._send_callback("step_progress", pct, fname)

                y_out = zansei_impl(
                    y,
                    sr,
                    m=int(self.params["m"]),
                    decay=float(self.params["decay"]),
                    pre_hp=float(self.params["pre_hp"]),
                    post_hp=float(self.params["post_hp"]),
                    filter_order=int(self.params["filter_order"]),
                    progress_cb=step_cb,
                    abort_cb=lambda: self._abort,  # 传入取消回调
                )

                # 保存（保持原格式 + 元数据）
                os.makedirs(self.output_dir, exist_ok=True)
                base, ext = os.path.splitext(fname)

                out_path = os.path.join(
                    self.output_dir,
                    f"{base}.{self.params['format'].lower() if self.params['format'] == 'flac' else 'm4a'}",
                )
                out_path = save_wav24_out(
                    in_path, y_out, sr, out_path, fmt=self.params["format"]
                )

                self._send_callback("log", f"文件已保存：{out_path}")
                self._send_callback("file_done", in_path, out_path)

            except Exception as e:
                err = "".join(traceback.format_exception_only(type(e), e)).strip()
                self._send_callback("error", fname, err)
                self._send_callback("log", f"[错误] {fname}: {err}")

            done += 1
            self._send_callback("overall_progress", done, total)
            self._send_callback("step_progress", 100, fname)

        self._send_callback("finished")

# ==================== GUI（tkinter实现） ====================
class MainWindow:
    def __init__(self, root):
        self.root = root
        self.root.title("DSRE v1.1.250908_beta")
        self.root.geometry("900x600")

        # 回调队列（用于线程间通信）
        self.callback_queue = queue.Queue()

        # 初始化UI组件
        self._init_ui()

        # 初始化工作线程
        self.worker: Optional[DSREWorker] = None

        # 初始化日志
        self.append_log("软件制作：屈乐凡")
        self.append_log("问题反馈：Le_Fan_Qv@outlook.com")
        self.append_log("交流群组：323861356（QQ）")

        # 启动回调处理循环
        self._process_callbacks()

    def _init_ui(self):
        # 主容器
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # 1. 左侧：文件列表
        left_frame = ttk.Frame(main_frame)
        left_frame.grid(row=0, column=0, rowspan=7, padx=5, pady=5, sticky="nsew")

        ttk.Label(left_frame, text="输入文件").pack(anchor=tk.CENTER)
        self.file_listbox = tk.Listbox(left_frame, width=30, height=20)
        self.file_listbox.pack(fill=tk.BOTH, expand=True, pady=5)

        # 2. 中间：操作按钮 + 输出目录
        mid_frame = ttk.Frame(main_frame)
        mid_frame.grid(row=0, column=1, rowspan=7, padx=5, pady=5, sticky="nsew")

        ttk.Label(mid_frame, text="操作").pack(anchor=tk.CENTER, pady=5)

        # 操作按钮
        ttk.Button(mid_frame, text="添加输入文件", command=self.on_add_files).pack(fill=tk.X, pady=2)
        ttk.Button(mid_frame, text="清空输入列表", command=self.on_clear_files).pack(fill=tk.X, pady=2)

        ttk.Label(mid_frame, text="输出目录").pack(anchor=tk.W, pady=10)
        self.outdir_var = tk.StringVar(value=os.path.abspath("output"))
        self.outdir_entry = ttk.Entry(mid_frame, textvariable=self.outdir_var)
        self.outdir_entry.pack(fill=tk.X, pady=2)
        ttk.Button(mid_frame, text="选择输出目录", command=self.on_choose_outdir).pack(fill=tk.X, pady=2)

        # 输出格式
        ttk.Label(mid_frame, text="输出编码格式").pack(anchor=tk.W, pady=10)
        self.format_var = tk.StringVar(value="ALAC")
        format_combo = ttk.Combobox(mid_frame, textvariable=self.format_var, values=["ALAC", "FLAC"], state="readonly")
        format_combo.pack(fill=tk.X, pady=2)

        # 控制按钮
        ttk.Label(mid_frame, text="控制").pack(anchor=tk.W, pady=10)
        self.start_btn = ttk.Button(mid_frame, text="开始处理", command=self.on_start)
        self.start_btn.pack(fill=tk.X, pady=2)
        self.cancel_btn = ttk.Button(mid_frame, text="取消处理", command=self.on_cancel, state=tk.DISABLED)
        self.cancel_btn.pack(fill=tk.X, pady=2)

        # 3. 右侧：参数设置 + 进度条
        right_frame = ttk.Frame(main_frame)
        right_frame.grid(row=0, column=2, rowspan=7, padx=5, pady=5, sticky="nsew")

        ttk.Label(right_frame, text="参数设置").pack(anchor=tk.CENTER, pady=5)

        # 参数表单
        param_frame = ttk.Frame(right_frame)
        param_frame.pack(fill=tk.X, pady=5)

        # 调制次数
        ttk.Label(param_frame, text="调制次数:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.m_var = tk.IntVar(value=8)
        m_spin = ttk.Spinbox(param_frame, from_=1, to=1024, textvariable=self.m_var, width=10)
        m_spin.grid(row=0, column=1, sticky=tk.W, pady=2)

        # 衰减幅度
        ttk.Label(param_frame, text="衰减幅度:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.decay_var = tk.DoubleVar(value=1.25)
        decay_spin = ttk.Spinbox(param_frame, from_=0.0, to=1024.0, increment=0.05, textvariable=self.decay_var, width=10)
        decay_spin.grid(row=1, column=1, sticky=tk.W, pady=2)

        # 预处理高通
        ttk.Label(param_frame, text="预处理高通(Hz):").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.pre_hp_var = tk.IntVar(value=3000)
        pre_hp_spin = ttk.Spinbox(param_frame, from_=1, to=384000, textvariable=self.pre_hp_var, width=10)
        pre_hp_spin.grid(row=2, column=1, sticky=tk.W, pady=2)

        # 后处理高通
        ttk.Label(param_frame, text="后处理高通(Hz):").grid(row=3, column=0, sticky=tk.W, pady=2)
        self.post_hp_var = tk.IntVar(value=16000)
        post_hp_spin = ttk.Spinbox(param_frame, from_=1, to=384000, textvariable=self.post_hp_var, width=10)
        post_hp_spin.grid(row=3, column=1, sticky=tk.W, pady=2)

        # 滤波器阶数
        ttk.Label(param_frame, text="滤波器阶数:").grid(row=4, column=0, sticky=tk.W, pady=2)
        self.order_var = tk.IntVar(value=11)
        order_spin = ttk.Spinbox(param_frame, from_=1, to=1000, textvariable=self.order_var, width=10)
        order_spin.grid(row=4, column=1, sticky=tk.W, pady=2)

        # 目标采样率
        ttk.Label(param_frame, text="目标采样率(Hz):").grid(row=5, column=0, sticky=tk.W, pady=2)
        self.sr_var = tk.IntVar(value=96000)
        sr_spin = ttk.Spinbox(param_frame, from_=1, to=384000, increment=1000, textvariable=self.sr_var, width=10)
        sr_spin.grid(row=5, column=1, sticky=tk.W, pady=2)

        # 进度条区域
        progress_frame = ttk.Frame(right_frame)
        progress_frame.pack(fill=tk.X, pady=20)

        ttk.Label(progress_frame, text="当前文件处理进度").pack(anchor=tk.W)
        self.file_progress = ttk.Progressbar(progress_frame, orient=tk.HORIZONTAL, length=200, mode='determinate')
        self.file_progress.pack(fill=tk.X, pady=2)

        ttk.Label(progress_frame, text="全部文件处理进度").pack(anchor=tk.W)
        self.all_progress = ttk.Progressbar(progress_frame, orient=tk.HORIZONTAL, length=200, mode='determinate')
        self.all_progress.pack(fill=tk.X, pady=2)

        # 当前状态标签
        self.status_label = ttk.Label(progress_frame, text="控制")
        self.status_label.pack(pady=10)

        # 4. 底部：日志区域
        ttk.Label(main_frame, text="日志").grid(row=7, column=0, columnspan=3, sticky=tk.W, pady=5)
        self.log_text = scrolledtext.ScrolledText(main_frame, height=10, state=tk.DISABLED)
        self.log_text.grid(row=8, column=0, columnspan=3, sticky="nsew")

        # 调整列权重
        main_frame.columnconfigure(0, weight=2)
        main_frame.columnconfigure(1, weight=1)
        main_frame.columnconfigure(2, weight=1)
        main_frame.rowconfigure(8, weight=1)

    def on_add_files(self):
        files = filedialog.askopenfilenames(
            title="选择输入文件",
            filetypes=[
                ("Audio Files", "*.wav *.mp3 *.m4a *.flac *.ogg *.aiff *.aif *.aac *.wma *.mka"),
                ("All Files", "*.*")
            ]
        )
        for f in files:
            if f and f not in self.file_listbox.get(0, tk.END):
                self.file_listbox.insert(tk.END, f)

    def on_clear_files(self):
        self.file_listbox.delete(0, tk.END)

    def on_choose_outdir(self):
        d = filedialog.askdirectory(title="选择输出目录", initialdir=self.outdir_var.get())
        if d:
            self.outdir_var.set(d)

    def get_params(self):
        return dict(
            m=self.m_var.get(),
            decay=self.decay_var.get(),
            pre_hp=self.pre_hp_var.get(),
            post_hp=self.post_hp_var.get(),
            target_sr=self.sr_var.get(),
            filter_order=self.order_var.get(),
            bit_depth=24,
            format=self.format_var.get()
        )

    def append_log(self, s: str):
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, s + "\n")
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)

    def on_start(self):
        files = self.file_listbox.get(0, tk.END)
        if not files:
            messagebox.showwarning("没有文件", "请至少添加一个输入文件")
            return
        outdir = self.outdir_var.get().strip() or os.path.abspath("output")

        # 重置进度
        self.file_progress["value"] = 0
        self.all_progress["value"] = 0
        self.status_label.config(text="正在初始化…")
        self.append_log(f"开始处理 {len(files)} 个文件…")

        # 禁用/启用按钮
        self.start_btn.config(state=tk.DISABLED)
        self.cancel_btn.config(state=tk.NORMAL)

        # 启动工作线程
        self.worker = DSREWorker(files, outdir, self.get_params(), self.callback_queue)
        self.worker.start()

    def on_cancel(self):
        if self.worker and self.worker.is_alive():
            self.append_log("正在取消…")
            self.worker.abort()

    def _process_callbacks(self):
        """处理后台线程的回调消息"""
        try:
            while True:
                callback_type, args = self.callback_queue.get_nowait()
                if callback_type == "log":
                    self.append_log(args[0])
                elif callback_type == "file_progress":
                    cur, total, fname = args
                    self.status_label.config(text=f"正在处理… [{cur}/{total}]: {fname}")
                    self.file_progress["value"] = 0
                elif callback_type == "step_progress":
                    pct, fname = args
                    self.file_progress["value"] = pct
                elif callback_type == "overall_progress":
                    done, total = args
                    pct = int(done * 100 / max(1, total))
                    self.all_progress["value"] = pct
                elif callback_type == "file_done":
                    in_path, out_path = args
                    self.append_log(f"处理完成: {os.path.basename(in_path)} -> {out_path}")
                elif callback_type == "error":
                    fname, err = args
                    self.append_log(f"[错误] {fname}: {err}")
                elif callback_type == "finished":
                    self.append_log("所有文件均已完成处理")
                    self.status_label.config(text="控制")
                    self.start_btn.config(state=tk.NORMAL)
                    self.cancel_btn.config(state=tk.DISABLED)
                    self.worker = None
        except queue.Empty:
            pass
        # 定时检查队列
        self.root.after(100, self._process_callbacks)

# ==================== 主函数 ====================
def main():
    root = tk.Tk()
    app = MainWindow(root)
    root.mainloop()

if __name__ == "__main__":
    main()
