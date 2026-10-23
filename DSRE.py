import os
import sys
import traceback
from typing import Optional

import subprocess
import soundfile as sf
import tempfile

import numpy as np
from scipy import signal
import librosa
import resampy

from PySide6 import QtCore, QtWidgets
from PySide6.QtGui import QIcon, QTextCursor

LANGUAGES = {
    "English": {
        "window_title": "DSRE v1.1.250908_beta",
        "lbl_input_files": "Input files (list of files added)",
        "btn_add": "Add input files",
        "btn_clear": "Clear input list",
        "lbl_out_dir": "Output directory",
        "btn_outdir": "Choose output folder",
        "lbl_ops": "Actions",
        "lbl_params": "Parameter settings",
        "row_m": "Modulation count",
        "row_decay": "Decay amount",
        "row_pre": "Pre-processing high-pass filter cutoff (Hz)",
        "row_post": "Post-processing high-pass filter cutoff (Hz)",
        "row_order": "Filter order",
        "row_sr": "Target sample rate (Hz)",
        "lbl_control": "Control",
        "btn_start": "Start processing",
        "btn_cancel": "Cancel processing",
        "lbl_prog_file": "Current file progress",
        "lbl_prog_all": "Overall progress",
        "lbl_format": "Output encoding format",
        "lbl_log": "Log",
        "footer_dev": "Developed by: 匠乐几",
        "footer_feedback": "Problem feedback: Le_Fan_Qv@outlook.com",
        "footer_group": "QQ Group: 323861356",
        "msg_init": "Initializing...",
        "msg_start": "Starting processing {} files...",
        "msg_done": "All files have been processed.",
        "msg_cancel": "Canceling...",
        "msg_error": "[Error]",
        "msg_file_done": "Finished",
        "lang_select": "Language"
    },
    "简体中文": {
        "window_title": "DSRE v1.1.250908_beta",
        "lbl_input_files": "输入文件",
        "btn_add": "添加输入文件",
        "btn_clear": "清空输入列表",
        "lbl_out_dir": "输出目录",
        "btn_outdir": "选择输出目录",
        "lbl_ops": "操作",
        "lbl_params": "参数设置",
        "row_m": "调制次数",
        "row_decay": "衰减幅度",
        "row_pre": "预处理高通滤波器截止频率(Hz)",
        "row_post": "后处理高通滤波器截止频率(Hz)",
        "row_order": "滤波器阶数",
        "row_sr": "目标采样率(Hz)",
        "lbl_control": "控制",
        "btn_start": "开始处理",
        "btn_cancel": "取消处理",
        "lbl_prog_file": "当前文件处理进度",
        "lbl_prog_all": "全部文件处理进度",
        "lbl_format": "输出编码格式",
        "lbl_log": "日志",
        "footer_dev": "软件制作：匠乐几",
        "footer_feedback": "问题反馈：Le_Fan_Qv@outlook.com",
        "footer_group": "交流群组：323861356（QQ）",
        "msg_init": "正在初始化…",
        "msg_start": "开始处理 {} 个文件…",
        "msg_done": "所有文件均已完成处理",
        "msg_cancel": "正在取消…",
        "msg_error": "[错误]",
        "msg_file_done": "处理完成",
        "lang_select": "语言"
    }
}

def add_ffmpeg_to_path():
    if hasattr(sys, "_MEIPASS"):  # 打包后的临时目录
        ffmpeg_dir = os.path.join(sys._MEIPASS, "ffmpeg")
    else:
        ffmpeg_dir = os.path.join(os.path.dirname(__file__), "ffmpeg")
    os.environ["PATH"] += os.pathsep + ffmpeg_dir

add_ffmpeg_to_path()

def save_wav24_out(in_path, y_out, sr, out_path, fmt="ALAC", normalize=True):
    import tempfile, subprocess, numpy as np, soundfile as sf, os

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

    # 临时 WAV 文件
    tmp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tmp_wav.close()
    sf.write(tmp_wav.name, data, sr, subtype="FLOAT")

    fmt = fmt.upper()
    out_path = os.path.splitext(out_path)[0] + (".m4a" if fmt == "ALAC" else ".flac")

    codec_map = {"ALAC": "alac", "FLAC": "flac"}
    sample_fmt_map = {"ALAC": "s32p", "FLAC": "s32"}  # 强制 24bit 整数

    if fmt == "ALAC":
        cmd = [
            "ffmpeg", "-y",
            "-i", tmp_wav.name,
            "-i", in_path,
            "-map", "0:a",       # 临时 WAV 音频
            "-map", "1:v?",      # 封面
            "-map_metadata", "1",# 元数据
            "-c:a", codec_map[fmt],
            "-sample_fmt", sample_fmt_map[fmt],
            "-c:v", "copy",
            out_path
        ]
    elif fmt == "FLAC":
        # 提取封面图片
        cover_tmp = None
        try:
            cover_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
            cover_tmp.close()
            subprocess.run(
                ["ffmpeg", "-y", "-i", in_path, "-an", "-c:v", "copy", cover_tmp.name],
                check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
        except Exception:
            cover_tmp = None

        if cover_tmp and os.path.exists(cover_tmp.name):
            cmd = [
                "ffmpeg", "-y",
                "-i", tmp_wav.name,  # WAV 音频
                "-i", in_path,       # 元数据来源
                "-i", cover_tmp.name, # 封面
                "-map", "0:a",       # 音频
                "-map", "2:v",       # 封面
                "-disposition:v", "attached_pic",
                "-map_metadata", "1",# 元数据
                "-c:a", codec_map[fmt],
                "-sample_fmt", sample_fmt_map[fmt],
                "-c:v", "copy",
                out_path
            ]
        else:
            cmd = [
                "ffmpeg", "-y",
                "-i", tmp_wav.name,
                "-i", in_path,
                "-map", "0:a",
                "-map_metadata", "1",
                "-c:a", codec_map[fmt],
                "-sample_fmt", sample_fmt_map[fmt],
                out_path
            ]

    subprocess.run(cmd, check=True)
    os.remove(tmp_wav.name)
    if fmt == "FLAC" and cover_tmp and os.path.exists(cover_tmp.name):
        os.remove(cover_tmp.name)

    return out_path

# ======== DSP：SSB 单边带频移 ========
def freq_shift_mono(x: np.ndarray, f_shift: float, d_sr: float) -> np.ndarray:
    N_orig = len(x)
    # pad 到 2 的幂次，便于 FFT/Hilbert 的实现效率
    N_padded = 1 << int(np.ceil(np.log2(max(1, N_orig))))
    S_hilbert = signal.hilbert(np.hstack((x, np.zeros(N_padded - N_orig, dtype=x.dtype))))
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
    b, a = signal.butter(filter_order, pre_hp / (sr / 2), 'highpass')
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
    b, a = signal.butter(filter_order, post_hp / (sr / 2), 'highpass')
    d_res = signal.filtfilt(b, a, d_res)

    adp_power = float(np.mean(np.abs(d_res)))
    src_power = float(np.mean(np.abs(x)))
    adj_factor = src_power / (adp_power + src_power + 1e-12)

    y = (x + d_res) * adj_factor
    return y

# ======== 后台工作线程 ========
class DSREWorker(QtCore.QThread):
    sig_log = QtCore.Signal(str)                         # 文本日志
    sig_file_progress = QtCore.Signal(int, int, str)     # 当前文件进度 (cur, total, filename)
    sig_step_progress = QtCore.Signal(int, str)          # 单文件内部进度(0~100), 文件名
    sig_overall_progress = QtCore.Signal(int, int)       # 总体进度 (done, total)
    sig_file_done = QtCore.Signal(str, str)              # 单文件完成 (in_path, out_path)
    sig_error = QtCore.Signal(str, str)                  # 错误 (filename, err_msg)
    sig_finished = QtCore.Signal()                       # 全部完成

    def __init__(self, files, output_dir, params, parent=None):
        super().__init__(parent)
        self.files = files
        self.output_dir = output_dir
        self.params = params
        self._abort = False

    def abort(self):
        self._abort = True

    def run(self):
        total = len(self.files)
        done = 0
        self.sig_overall_progress.emit(done, total)

        for idx, in_path in enumerate(self.files, start=1):
            if self._abort:
                break

            fname = os.path.basename(in_path)
            self.sig_file_progress.emit(idx, total, fname)
            self.sig_step_progress.emit(0, fname)

            try:
                # 读取
                self.sig_log.emit(f"Loading: {in_path}")
                y, sr = librosa.load(in_path, mono=False, sr=None)

                # 对齐为 (ch, n)
                if y.ndim == 1:
                    y = y[np.newaxis, :]
                # 重采样
                target_sr = int(self.params["target_sr"])
                if sr != target_sr:
                    self.sig_log.emit(f"Processing: {fname}: {sr} -> {target_sr}")
                    y = resampy.resample(y, sr, target_sr, filter='kaiser_fast')
                    sr = target_sr

                # 处理
                def step_cb(cur, m):
                    pct = int(cur * 100 / max(1, m))
                    self.sig_step_progress.emit(pct, fname)

                y_out = zansei_impl(
                    y, sr,
                    m=int(self.params["m"]),
                    decay=float(self.params["decay"]),
                    pre_hp=float(self.params["pre_hp"]),
                    post_hp=float(self.params["post_hp"]),
                    filter_order=int(self.params["filter_order"]),
                    progress_cb=step_cb,
                    abort_cb=lambda: self._abort  # 传入取消回调
                )

                # 保存（保持原格式 + 元数据）
                os.makedirs(self.output_dir, exist_ok=True)
                base, ext = os.path.splitext(fname)

                out_path = os.path.join(self.output_dir,
                                        f"{base}.{self.params['format'].lower() if self.params['format'] == 'flac' else 'm4a'}")
                out_path = save_wav24_out(in_path, y_out, sr, out_path, fmt=self.params['format'])

                self.sig_log.emit(f"Saved: {out_path}")
                self.sig_file_done.emit(in_path, out_path)

            except Exception as e:
                err = "".join(traceback.format_exception_only(type(e), e)).strip()
                self.sig_error.emit(fname, err)
                self.sig_log.emit(f"[Error] {fname}: {err}")

            done += 1
            self.sig_overall_progress.emit(done, total)
            self.sig_step_progress.emit(100, fname)

        self.sig_finished.emit()

# ======== GUI ========
class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.current_lang = "English"

        # 获取相对路径的图标
        icon_path = os.path.join(os.path.dirname(__file__), "logo.ico")
        self.setWindowIcon(QIcon(icon_path))

        self.resize(900, 600)

        # 文件列表
        self.list_files = QtWidgets.QListWidget()
        self.btn_add = QtWidgets.QPushButton()
        self.btn_clear = QtWidgets.QPushButton()
        self.btn_outdir = QtWidgets.QPushButton()
        self.le_outdir = QtWidgets.QLineEdit()
        self.le_outdir.setText(os.path.abspath("output"))

        # 参数
        self.sb_m = QtWidgets.QSpinBox()
        self.sb_m.setRange(1, 1024)
        self.sb_m.setValue(8)
        self.dsb_decay = QtWidgets.QDoubleSpinBox()
        self.dsb_decay.setRange(0.0, 1024)
        self.dsb_decay.setSingleStep(0.05)
        self.dsb_decay.setValue(1.25)
        self.sb_pre = QtWidgets.QSpinBox()
        self.sb_pre.setRange(1, 384000)
        self.sb_pre.setValue(3000)
        self.sb_post = QtWidgets.QSpinBox()
        self.sb_post.setRange(1, 384000)
        self.sb_post.setValue(16000)
        self.sb_order = QtWidgets.QSpinBox()
        self.sb_order.setRange(1, 1000)
        self.sb_order.setValue(11)
        self.sb_sr = QtWidgets.QSpinBox()
        self.sb_sr.setRange(1, 384000)
        self.sb_sr.setSingleStep(1000)
        self.sb_sr.setValue(96000)

        # 进度
        self.pb_file = QtWidgets.QProgressBar()    # 单文件进度
        self.pb_all = QtWidgets.QProgressBar()     # 全部进度
        self.lbl_now = QtWidgets.QLabel()

        # 控制按钮
        self.btn_start = QtWidgets.QPushButton()
        self.btn_cancel = QtWidgets.QPushButton()
        self.btn_cancel.setEnabled(False)

        # 日志
        self.te_log = QtWidgets.QTextEdit()
        self.te_log.setReadOnly(True)

        # 语言选择器
        self.cb_lang = QtWidgets.QComboBox()
        self.cb_lang.addItems(["English", "简体中文"])
        self.cb_lang.currentTextChanged.connect(self.change_language)

        # 动态文本标签引用
        self.lbl_files_title = QtWidgets.QLabel()
        self.lbl_ops_title = QtWidgets.QLabel()
        self.lbl_outdir_title = QtWidgets.QLabel()
        self.lbl_format_title = QtWidgets.QLabel()
        self.lbl_params_title = QtWidgets.QLabel()
        self.lbl_prog_file_title = QtWidgets.QLabel()
        self.lbl_prog_all_title = QtWidgets.QLabel()
        self.lbl_log_title = QtWidgets.QLabel()
        self.lbl_lang_title = QtWidgets.QLabel()

        # ===== 布局 =====
        grid = QtWidgets.QGridLayout()

        # === 左列：输入文件 ===
        vleft = QtWidgets.QVBoxLayout()
        self.lbl_files_title.setAlignment(QtCore.Qt.AlignHCenter)
        vleft.addWidget(self.lbl_files_title)
        vleft.addWidget(self.list_files)
        grid.addLayout(vleft, 0, 0, 7, 1)

        # === 中列：操作 ===
        vmid = QtWidgets.QVBoxLayout()
        self.lbl_ops_title.setAlignment(QtCore.Qt.AlignHCenter)
        vmid.addWidget(self.lbl_ops_title)

        vbtn = QtWidgets.QVBoxLayout()
        vbtn.addWidget(self.btn_add)
        vbtn.addWidget(self.btn_clear)
        vbtn.addSpacing(10)
        vbtn.addWidget(self.lbl_outdir_title)
        vbtn.addWidget(self.le_outdir)
        vbtn.addWidget(self.btn_outdir)
        vbtn.addSpacing(10)
        
        # 语言控件位置
        vbtn.addWidget(self.lbl_lang_title)
        vbtn.addWidget(self.cb_lang)
        vbtn.addSpacing(20)

        # 把 lbl_now ("控制") 放在这里
        vbtn.addWidget(self.lbl_now)

        vbtn.addWidget(self.btn_start)
        vbtn.addWidget(self.btn_cancel)
        vbtn.addStretch(1)

        # 输出格式选择
        self.cb_format = QtWidgets.QComboBox()
        self.cb_format.addItems(["ALAC", "FLAC"])  # 两种可选格式
        vbtn.addWidget(self.lbl_format_title)
        vbtn.addWidget(self.cb_format)

        vmid.addLayout(vbtn)
        grid.addLayout(vmid, 0, 1, 7, 1)

        # === 右列：参数设置 + 进度 ===
        vright = QtWidgets.QVBoxLayout()
        self.lbl_params_title.setAlignment(QtCore.Qt.AlignHCenter)
        vright.addWidget(self.lbl_params_title)

        self.form = QtWidgets.QFormLayout()
        vright.addLayout(self.form)

        vright.addSpacing(20)

        vprog = QtWidgets.QVBoxLayout()
        vprog.addWidget(self.lbl_prog_file_title)
        vprog.addWidget(self.pb_file)
        vprog.addWidget(self.lbl_prog_all_title)
        vprog.addWidget(self.pb_all)
        vprog.addStretch(1)
        vright.addLayout(vprog)
        grid.addLayout(vright, 0, 2, 7, 1)

        # === 底部日志 ===
        grid.addWidget(self.lbl_log_title, 7, 0)
        grid.addWidget(self.te_log, 8, 0, 1, 3)

        self.setLayout(grid)

        # 连接信号
        self.btn_add.clicked.connect(self.on_add_files)
        self.btn_clear.clicked.connect(self.list_files.clear)
        self.btn_outdir.clicked.connect(self.on_choose_outdir)
        self.btn_start.clicked.connect(self.on_start)
        self.btn_cancel.clicked.connect(self.on_cancel)

        self.worker: Optional[DSREWorker] = None
        
        # 初始化 UI 翻译
        self.retranslate_ui()
        self.append_log(LANGUAGES[self.current_lang]["footer_dev"])
        self.append_log(LANGUAGES[self.current_lang]["footer_feedback"])
        self.append_log(LANGUAGES[self.current_lang]["footer_group"])

    def retranslate_ui(self):
        trans = LANGUAGES[self.current_lang]
        self.setWindowTitle(trans["window_title"])
        self.lbl_files_title.setText(trans["lbl_input_files"])
        self.btn_add.setText(trans["btn_add"])
        self.btn_clear.setText(trans["btn_clear"])
        self.lbl_outdir_title.setText(trans["lbl_out_dir"])
        self.btn_outdir.setText(trans["btn_outdir"])
        self.lbl_ops_title.setText(trans["lbl_ops"])
        self.lbl_now.setText(trans["lbl_control"])
        self.btn_start.setText(trans["btn_start"])
        self.btn_cancel.setText(trans["btn_cancel"])
        self.lbl_format_title.setText(trans["lbl_format"])
        self.lbl_params_title.setText(trans["lbl_params"])
        self.lbl_prog_file_title.setText(trans["lbl_prog_file"])
        self.lbl_prog_all_title.setText(trans["lbl_prog_all"])
        self.lbl_log_title.setText(trans["lbl_log"])
        self.lbl_lang_title.setText(trans["lang_select"])

        # 更新 Form 布局中的标签
        for i in reversed(range(self.form.count())): 
            self.form.itemAt(i).widget().setParent(None)
            
        self.form.addRow(trans["row_m"], self.sb_m)
        self.form.addRow(trans["row_decay"], self.dsb_decay)
        self.form.addRow(trans["row_pre"], self.sb_pre)
        self.form.addRow(trans["row_post"], self.sb_post)
        self.form.addRow(trans["row_order"], self.sb_order)
        self.form.addRow(trans["row_sr"], self.sb_sr)

    def change_language(self, lang):
        self.current_lang = lang
        self.retranslate_ui()

    def on_add_files(self):
        filters = (
            "Audio Files (*.wav *.mp3 *.m4a *.flac *.ogg *.aiff *.aif *.aac *.wma *.mka);;"
            "All Files (*.*)"
        )
        files, _ = QtWidgets.QFileDialog.getOpenFileNames(self, "Select input files", "", filters)
        for f in files:
            if f and (self.list_files.findItems(f, QtCore.Qt.MatchFlag.MatchExactly) == []):
                self.list_files.addItem(f)

    def on_choose_outdir(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Select output directory", self.le_outdir.text() or "")
        if d:
            self.le_outdir.setText(d)

    def params(self):
        return dict(
            m=self.sb_m.value(),
            decay=self.dsb_decay.value(),
            pre_hp=self.sb_pre.value(),
            post_hp=self.sb_post.value(),
            target_sr=self.sb_sr.value(),
            filter_order=self.sb_order.value(),
            bit_depth=24,  # 固定输出 24bit
            format=self.cb_format.currentText()  # ALAC 或 FLAC
        )

    def append_log(self, s: str):
        self.te_log.append(s)
        self.te_log.moveCursor(QTextCursor.End)

    def on_start(self):
        trans = LANGUAGES[self.current_lang]
        files = [self.list_files.item(i).text() for i in range(self.list_files.count())]
        if not files:
            QtWidgets.QMessageBox.warning(self, "Warning", trans["lbl_input_files"])
            return
        outdir = self.le_outdir.text().strip() or os.path.abspath("output")

        # 置零进度
        self.pb_all.setValue(0)
        self.pb_file.setValue(0)
        self.lbl_now.setText(trans["msg_init"])
        self.append_log(trans["msg_start"].format(len(files)))

        # 锁定按钮
        self.btn_start.setEnabled(False)
        self.btn_cancel.setEnabled(True)

        # 启动后台线程
        self.worker = DSREWorker(files, outdir, self.params())
        self.worker.sig_log.connect(self.append_log)
        self.worker.sig_file_progress.connect(self.on_file_progress)
        self.worker.sig_step_progress.connect(self.on_step_progress)
        self.worker.sig_overall_progress.connect(self.on_overall_progress)
        self.worker.sig_file_done.connect(self.on_file_done)
        self.worker.sig_error.connect(self.on_error)
        self.worker.sig_finished.connect(self.on_finished)
        self.worker.start()

    @QtCore.Slot(int, int, str)
    def on_file_progress(self, cur, total, fname):
        self.lbl_now.setText(f"Processing [{cur}/{total}]: {fname}")
        self.pb_file.setValue(0)

    @QtCore.Slot(int, str)
    def on_step_progress(self, pct, fname):
        self.pb_file.setValue(pct)

    @QtCore.Slot(int, int)
    def on_overall_progress(self, done, total):
        pct = int(done * 100 / max(1, total))
        self.pb_all.setValue(pct)

    @QtCore.Slot(str, str)
    def on_file_done(self, in_path, out_path):
        trans = LANGUAGES[self.current_lang]
        self.append_log(f"{trans['msg_file_done']}: {os.path.basename(in_path)} -> {out_path}")

    @QtCore.Slot(str, str)
    def on_error(self, fname, err):
        trans = LANGUAGES[self.current_lang]
        self.append_log(f"{trans['msg_error']} {fname}: {err}")

    def on_cancel(self):
        if self.worker and self.worker.isRunning():
            self.append_log(LANGUAGES[self.current_lang]["msg_cancel"])
            self.worker.abort()

    def on_finished(self):
        trans = LANGUAGES[self.current_lang]
        self.append_log(trans["msg_done"])
        self.lbl_now.setText(trans["lbl_control"])
        self.btn_start.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.worker = None

def main():

    import ctypes
    myappid = "com.lefanqv.dsre"  # 你自定义的应用 ID，必须是字符串
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)

    app = QtWidgets.QApplication(sys.argv)

    # 全局设置应用图标
    icon_path = os.path.join(os.path.dirname(__file__), "logo.ico")
    app.setWindowIcon(QIcon(icon_path))

    w = MainWindow()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
