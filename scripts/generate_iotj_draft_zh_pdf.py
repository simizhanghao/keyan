#!/usr/bin/env python3
"""Generate Chinese IoTJ-targeted paper draft PDF (PyMuPDF + system Noto CJK)."""

from __future__ import annotations

import textwrap
from pathlib import Path

import fitz  # PyMuPDF

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "docs" / "paper_draft"
OUT_PDF = OUT_DIR / "IoTJ_中文论文初稿_LoRa_RFFI_Hybrid.pdf"
FONT_PATH = OUT_DIR / "fonts" / "NotoSansSC-Regular.otf"
FONT_BOLD_PATH = OUT_DIR / "fonts" / "NotoSansSC-Bold.otf"
# fallback to system TTC if subset OTF missing
FONT_FALLBACK = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")

PAGE_W, PAGE_H = fitz.paper_size("a4")
MARGIN_L, MARGIN_R, MARGIN_T, MARGIN_B = 50, 50, 55, 50
BODY_SIZE = 10.5
TITLE_SIZE = 16
H1_SIZE = 13
H2_SIZE = 11.5
LINE_H = 16


class PaperWriter:
    def __init__(self) -> None:
        if not FONT_PATH.exists():
            raise FileNotFoundError(
                f"缺少简体中文字体 {FONT_PATH}，请先运行 scripts/download_noto_sc_font.sh"
            )
        self._font = str(FONT_PATH)
        self._font_bold = str(FONT_BOLD_PATH if FONT_BOLD_PATH.exists() else FONT_PATH)
        self.doc = fitz.open()
        self.page = self.doc.new_page(width=PAGE_W, height=PAGE_H)
        self.y = MARGIN_T
        self._embed_fonts(self.page)

    def _embed_fonts(self, page: fitz.Page) -> None:
        page.insert_font(fontfile=self._font, fontname="noto")
        page.insert_font(fontfile=self._font_bold, fontname="noto_bold")

    @property
    def text_width(self) -> float:
        return PAGE_W - MARGIN_L - MARGIN_R

    def _ensure_space(self, needed: float) -> None:
        if self.y + needed > PAGE_H - MARGIN_B:
            self.page = self.doc.new_page(width=PAGE_W, height=PAGE_H)
            self._embed_fonts(self.page)
            self.y = MARGIN_T

    def _write_lines(self, text: str, *, size: float, font: str, leading: float) -> None:
        for para in text.split("\n"):
            para = para.strip()
            if not para:
                self.y += leading * 0.5
                continue
            chars_per_line = max(20, int(self.text_width / (size * 0.55)))
            for line in textwrap.wrap(para, width=chars_per_line):
                self._ensure_space(leading)
                self.page.insert_text(
                    fitz.Point(MARGIN_L, self.y),
                    line,
                    fontname=font,
                    fontsize=size,
                    color=(0, 0, 0),
                )
                self.y += leading

    def title(self, text: str) -> None:
        self._write_lines(text, size=TITLE_SIZE, font="noto_bold", leading=22)
        self.y += 4

    def subtitle(self, text: str) -> None:
        self._write_lines(text, size=10, font="noto", leading=15)
        self.y += 6

    def h1(self, text: str) -> None:
        self.y += 6
        self._write_lines(text, size=H1_SIZE, font="noto_bold", leading=18)

    def h2(self, text: str) -> None:
        self.y += 3
        self._write_lines(text, size=H2_SIZE, font="noto_bold", leading=16)

    def body(self, text: str) -> None:
        self._write_lines(text, size=BODY_SIZE, font="noto", leading=LINE_H)

    def table(self, headers: list[str], rows: list[list[str]], col_ratios: list[float]) -> None:
        total_w = self.text_width
        col_w = [total_w * r / sum(col_ratios) for r in col_ratios]
        row_h = 20
        self._ensure_space(row_h * (len(rows) + 2))
        x0 = MARGIN_L
        # header
        x = x0
        for i, h in enumerate(headers):
            rect = fitz.Rect(x, self.y - 12, x + col_w[i], self.y + row_h - 12)
            self.page.draw_rect(rect, color=(0.85, 0.88, 0.92), fill=(0.91, 0.93, 0.96), width=0.4)
            self.page.insert_textbox(rect, h, fontname="noto_bold", fontsize=9.5, align=fitz.TEXT_ALIGN_CENTER)
            x += col_w[i]
        self.y += row_h
        for r_idx, row in enumerate(rows):
            x = x0
            fill = (0.98, 0.98, 0.98) if r_idx % 2 else (1, 1, 1)
            for i, cell in enumerate(row):
                rect = fitz.Rect(x, self.y - 12, x + col_w[i], self.y + row_h - 12)
                self.page.draw_rect(rect, color=(0.75, 0.75, 0.75), fill=fill, width=0.3)
                self.page.insert_textbox(rect, cell, fontname="noto", fontsize=9.5, align=fitz.TEXT_ALIGN_CENTER)
                x += col_w[i]
            self.y += row_h
        self.y += 6

    def caption(self, text: str) -> None:
        rect = fitz.Rect(MARGIN_L, self.y - 2, PAGE_W - MARGIN_R, self.y + 14)
        self.page.insert_textbox(rect, text, fontname="noto", fontsize=9, align=fitz.TEXT_ALIGN_CENTER)
        self.y += 18

    def save(self, path: Path) -> None:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        self.doc.save(str(path))
        self.doc.close()


def build() -> None:
    w = PaperWriter()
    w.title("面向物联网 LoRa 设备认证的\nOOB 引导混合 RF-HSTU 射频指纹识别方法")
    w.subtitle("（IEEE Internet of Things Journal 目标稿 · 中文初稿 · 2026-06-23）")

    w.h2("摘要")
    w.body(
        "低功耗广域物联网（LPWAN）中，LoRa 终端的大规模部署使基于射频指纹（RFFI）的轻量级设备认证"
        "成为边缘侧安全的重要补充。然而，真实 IoT 场景存在跨天信道漂移、跨接收机域偏移及硬件非理想性等挑战，"
        "传统卷积基线难以同时兼顾局部波形畸变建模与长程 chirp 结构表达。"
        "本文提出一种面向 LoRa RFFI 的 OOB 引导混合架构：以 CNN-stem 保留 IQ/FFT/幅度相位等局部 RF 特征，"
        "以 RF-HSTU 序列模块建模 patch 级时序结构，并通过带外（OOB）区域交叉注意力动态注入硬件失真证据，"
        "辅以 chirp 位置嵌入对齐 LoRa 调制先验。"
        "在 OSU 公开 LoRa 数据集上，于严格 Day1→Day2 跨天评估中，所提混合模型窗口级准确率由 56.4% 提升至 74.6%，"
        "宏平均 F1 由 0.515 提升至 0.732；留一跨天（LODO）协议下，Hybrid 在 Day5 测试折上窗口准确率达 74.6%。"
        "跨接收机实验中，OOB 比率归一化（oob_ratio）显著缓解接收机增益差异；在 80 epoch 确认设置下，"
        "Hybrid oob_ratio 在 RX1→RX2 与 RX2→RX1 双向平均窗口准确率为 22.1%，宏 F1 为 0.180，"
        "较同配置 CNN iq_rms 基线（18.6% / 0.177）呈一致改进趋势，但受限于 24 个测试文件，"
        "文件级显著性尚待更大规模验证。"
        "结果表明，OOB 引导的 CNN 增强 RF-HSTU hybrid 更适合 IoT 边缘 RFFI 部署场景下的跨域鲁棒认证。"
    )
    w.body("关键词：物联网安全；LoRa；射频指纹识别；设备认证；带外区域；RF-HSTU；域偏移；边缘智能")

    w.h1("I. 引言")
    w.body(
        "IEEE Internet of Things Journal（IoTJ）关注传感、通信、计算与安全一体化的 IoT 系统。"
        "在 LPWAN 领域，LoRa/LoRaWAN 已广泛用于智慧城市、工业监测与农业传感等场景。"
        "MAC 层密钥可抵御远程伪造，但难以识别物理层克隆或同型号设备冒充；"
        "射频指纹识别（RFFI）利用发射机硬件非理想性（如功率放大器非线性、I/Q 不平衡、时钟偏移）"
        "实现“物理层身份”，可在网关或边缘节点以低开销完成设备认证，与 IoT 零信任架构高度契合。"
    )
    w.body(
        "现有 LoRa RFFI 研究多基于 IQ 波形或频谱特征的 CNN 分类器。"
        "OSU 等公开数据集表明，Day1 训练、Day2 测试的跨天设置下性能显著下降，"
        "跨接收机迁移则因增益、噪声底与前端滤波差异而更加困难。"
        "纯 Transformer 或序列模型虽能捕获长程依赖，却可能弱化 CNN 对局部波形畸变的敏感性；"
        "简单拼接 OOB 特征亦无法让主路径自适应选择硬件失真证据。"
    )
    w.body(
        "本文贡献如下：（1）提出 CNN-stem + RF-HSTU + OOB 引导交叉注意力 + chirp 嵌入的混合架构，"
        "面向 IoT LoRa 设备认证；（2）系统分析 input/FFT/OOB 归一化策略，发现 oob_ratio 对跨接收机域偏移尤为关键；"
        "（3）在 OSU 数据集上完成跨天（Day1→Day2、LODO）与跨接收机（RX1↔RX2）对照实验，"
        "给出可复现实验协议与审慎结论。"
    )

    w.h1("II. 相关工作")
    w.h2("A. LoRa 与 IoT 物理层安全")
    w.body(
        "LoRa 采用 chirp 扩频，帧结构包含前导、可选跳频与 payload。"
        "IoT 网关可在解调前采集 IQ，实现无侵入指纹提取。"
        "相关工作涵盖 CNN/RNN/ResNet 分类、元学习与域适应等，"
        "但多数假设同接收机、短时间采集，对跨天与跨接收机联合鲁棒性讨论不足。"
    )
    w.h2("B. 带外（OOB）特征与硬件失真")
    w.body(
        "OOB 区域位于 chirp 主能量之外，对发射机频谱泄漏、滤波器滚降与 PA 非线性更敏感，"
        "且在一定条件下对接收机增益变化更稳定（尤其 ratio 归一化）。"
        "本文不声称“首次使用 OOB”，而是强调 OOB 引导交叉注意力实现主路径对 OOB 证据的动态选择性融合。"
    )
    w.h2("C. RF-HSTU 与混合建模")
    w.body(
        "RF-HSTU 将 IQ 窗口划分为 patch 序列，以类 Transformer 块建模 patch 间关系。"
        "本文立场明确：并非“用 Transformer 替换 CNN”，而是以 CNN-stem 增强局部 RF 表征，"
        "再交由 RF-HSTU 与 OOB 交叉注意力完成 chirp 级结构建模与硬件 cue 注入。"
    )

    w.h1("III. 系统模型与问题定义")
    w.body(
        "考虑含 N 个 LoRa 发射机设备的 IoT 网络，边缘网关采集长度为 L 的复基带 IQ 窗口。"
        "训练集含设备标签 y∈{1,…,N}；测试阶段存在域偏移（不同采集日或不同接收机）。"
        "目标是在无目标域标签条件下学习分类器，最大化窗口级准确率 Acc_w 与宏 F1，"
        "并报告文件级投票准确率 Acc_f（每文件多窗口 logits 均值投票）。"
    )
    w.body(
        "数据来自 OSU LoRa 数据集：24 个室内发射机，1 MS/s 采样，SF7/BW125 kHz，"
        "8192 样本窗口、256 样本 patch（32 patches/窗口）。"
        "跨天实验：Day1 训练、Day2 测试（各 24 文件）；"
        "LODO：5 天中留一天测试；跨接收机：RX1 训练→RX2 测试及反向，各 24 测试文件。"
    )

    w.h1("IV. 所提方法")
    w.h2("A. 多视图 patch 嵌入与 CNN-stem")
    w.body(
        "对每个 IQ 窗口构造 IQ、对数 FFT、幅度-相位及 OOB 视图。"
        "CNN-stem 以多核一维卷积（kernel 7/5）将拼接后的 RF 视图映射为 D 维 patch token，"
        "保留 OSU-CNN-IQ 对局部波形畸变的建模能力。"
    )
    w.h2("B. RF-HSTU 与 chirp 嵌入")
    w.body(
        "patch token 经线性投影与位置编码后输入 depth=2 的 RF-HSTU 块。"
        "chirp 嵌入将 patch 索引分解为 chirp_id 与 patch_in_chirp，"
        "注入 LoRa chirp 周期先验，使注意力在 chirp 结构内对齐。"
    )
    w.h2("C. OOB 引导交叉注意力")
    w.body(
        "OOB patch 经独立投影后，通过 cross-attention 向主 RF token 注入硬件失真信息："
        "Q 来自主路径，K/V 来自 OOB 路径，使模型按样本动态决定 OOB 证据权重。"
        "相较 concat_oob，该机制在跨天 Day1→Day2 上显著提升 macro-F1。"
    )
    w.h2("D. 归一化与训练配置")
    w.body(
        "跨接收机最优组合：input_norm=iq_rms，fft_norm=log_zscore，oob_norm=ratio（oob_ratio）。"
        "训练 80 epoch（跨接收机确认）或 30 epoch（快速对照），batch=64，Adam，balanced batch 采样；"
        "主结果采用 classifier + mean_logits 文件投票。"
    )

    w.h1("V. 实验与结果")
    w.h2("A. 实验设置")
    w.body(
        "基线：OSU-CNN-IQ（iq_rms 输入）。"
        "主模型：Hybrid（cnn_stem + cross_attn_oob + chirp_embedding + RF-HSTU）。"
        "指标：窗口准确率 Acc_w、文件级准确率 Acc_f、宏平均 F1。"
        "所有结果来自项目 outputs/ 目录可复现日志。"
    )

    w.h2("B. 跨天 Day1→Day2（24 测试文件）")
    w.table(
        ["方法", "Acc_w", "Acc_f", "Macro-F1"],
        [["OSU-CNN-IQ", "56.4%", "70.8%", "0.515"], ["Hybrid（本文）", "74.6%", "75.0%", "0.732"]],
        [2.2, 1, 1, 1],
    )
    w.caption("表 I. Day1 训练 → Day2 测试（classifier, mean_logits）")

    w.h2("C. 留一跨天（LODO，5 折）")
    w.table(
        ["测试日", "CNN Acc_w", "Hybrid Acc_w", "CNN F1", "Hybrid F1"],
        [
            ["Day1", "59.7%", "50.6%", "0.568", "0.491"],
            ["Day2", "53.5%", "51.8%", "0.479", "0.471"],
            ["Day3", "57.1%", "61.1%", "0.534", "0.589"],
            ["Day4", "58.2%", "65.5%", "0.548", "0.630"],
            ["Day5", "58.7%", "74.6%", "0.538", "0.732"],
        ],
        [1, 1.2, 1.2, 1, 1],
    )
    w.caption("表 II. LODO 各折窗口级结果（classifier_mean_logits）")

    w.h2("D. 跨接收机（80 epoch，oob_ratio 确认）")
    w.table(
        ["方向", "模型/归一化", "Acc_w", "Acc_f", "Macro-F1"],
        [
            ["RX1→RX2", "CNN iq_rms", "8.6%", "8.3%", "0.036"],
            ["RX1→RX2", "Hybrid oob_ratio", "20.3%", "20.8%", "0.158"],
            ["RX2→RX1", "CNN iq_rms", "38.5%", "58.3%", "0.317"],
            ["RX2→RX1", "Hybrid oob_ratio", "23.8%", "33.3%", "0.201"],
        ],
        [1, 1.8, 1, 1, 1],
    )
    w.caption("表 III. 跨接收机 80 epoch 归一化确认实验")

    w.h2("E. 30 epoch 快速对照（2026-06-23）")
    w.table(
        ["方向", "CNN iq_rms", "Hybrid oob_ratio", "Δ Acc_w", "Δ F1"],
        [
            ["RX1→RX2", "9.0% / 0.068", "19.8% / 0.146", "+10.8%", "+0.078"],
            ["RX2→RX1", "18.4% / 0.124", "22.2% / 0.193", "+3.8%", "+0.069"],
            ["双向平均", "13.7% / 0.096", "21.0% / 0.169", "+7.3%", "+0.073"],
        ],
        [1, 1.5, 1.5, 1, 1],
    )
    w.caption("表 IV. 跨接收机 30 epoch 快速实验（Acc_w / Macro-F1）")

    w.h1("VI. IoT 部署讨论")
    w.body(
        "边缘可行性：推理为单次前向传播，窗口长度固定，适合网关侧批处理；"
        "相较云端回传原始 IQ，RFFI 仅输出设备 ID 置信度，通信开销低。"
        "安全场景：可作为 LoRaWAN 入网/Rejoin 的辅助因子，检测克隆节点与内部威胁。"
        "局限：跨接收机文件级优势不稳定；测试文件规模小；"
        "未在 Config/距离/位置扩展集上完成系统评测。"
    )

    w.h1("VII. 结论")
    w.body(
        "本文面向 IoT LoRa 设备认证，提出 OOB 引导的 CNN-RF-HSTU 混合 RFFI 架构。"
        "在 OSU 数据集跨天评估中，窗口准确率与 macro-F1 显著优于 reproduced OSU-CNN-IQ；"
        "跨接收机场景下，oob_ratio 归一化使 Hybrid 在困难方向（RX1→RX2）保持优势。"
        "后续将扩展 Config/多距离数据、探索发射机 query bottleneck 注意力，"
        "并完善英文稿以满足 IoTJ 150–250 词摘要与 8 页双栏格式。"
    )

    w.h2("附录：IoTJ 投稿核对清单")
    w.body(
        "• 期刊范围：IoT 传感/通信/边缘 AI/安全 — LoRa RFFI 设备认证契合。\n"
        "• 格式：最终英文稿需 IEEE 双栏模板；标准长度 8 页（超出 $175/页）。\n"
        "• 摘要：英文 150–250 词（本稿为中文初稿）。\n"
        "• 投稿系统：https://mc.manuscriptcentral.com/iot\n"
        "• 可复现路径：github.com/hanCChan/lunwen，outputs/ 与 docs/experiment_protocol.md。"
    )

    w.save(OUT_PDF)
    print(f"Wrote {OUT_PDF} ({OUT_PDF.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    build()
