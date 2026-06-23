# OSU LoRa 数据扩展计划

> **状态**：规划文档，**不启动下载、不训练、不改模型**。  
> **依据**：OSU NetSTAR 官方 release（[LoRa-Dataset 索引](https://research.engr.oregonstate.edu/hamdaoui/RFFP-dataset/LoRa-Dataset/)）、IEEE Network 2022 / IEEE Access 2021 场景描述、本地 `data/raw/osu_lora/` 盘点（2026-06-16）。

---

## 0. 官方 setup 总览

OSU LoRa RFFI 数据集在 release 中共 **7 个顶层 setup**（约 1.2TB，含 IQ + FFT + SigMF meta）：

| # | 官方目录 | 子目录 | 本地 |
|---|----------|--------|------|
| 1 | `Diff_Days_Indoor_Setup` | Day1–Day5 × Device1–25 × IQ_1..10 | **部分** |
| 2 | `Diff_Days_Outdoor_Setup` | 同上 | 未下载 |
| 3 | `Diff_Days_Wired_Setup` | Day1–Day5 × Device1–25 × 多 RX 文件 | 未下载 |
| 4 | `Diff_Configurations_Setup` | Config1–4 × IQ_1..25（flat） | 未下载 |
| 5 | `Diff_Locations_Setup` | Location1–3 × IQ_1..25（flat） | 未下载 |
| 6 | `Diff_Distances_Setup` | 5m/10m/15m/20m × IQ_1..25（flat） | 未下载 |
| 7 | `Diff_Receivers_Setup` | 6 个子场景 × 2 接收机 | **仅 Indoor SameTx** |

**统一实验口径（本项目已固定）**

- 25 个 Pycom 发射机中 **排除 raw Device9**（官方 Day2/Device9 无可用 IQ；见 `aria2_missing_day2_device9.txt`）。
- 重映射为 **24 类**：`device=1..24`，`label=0..23`（与 `generate_manifest_days.py` / `generate_manifest_receivers.py` 一致）。
- 优先下载 **IQ `.dat`**；FFT 可后续按需补（CNN-FFT 对照或 ablation）。

---

## 1. 已下载数据

### 1.1 Diff_Days（Indoor）

| 项 | 状态 |
|----|------|
| 路径 | `data/raw/osu_lora/Diff_Days_Indoor_Setup/` |
| 天数 | Day1–Day5 目录齐全 |
| 文件 | 每 device 仅 **`IQ_1.dat`**（+ 对应 `.sigmf-meta`） |
| 规模 | **124 / 125** 个 IQ 文件（缺 `Day2/Device9/IQ_1.dat`；目录存在但为空） |
| 未下 | 每 device 的 **`IQ_2.dat` … `IQ_10.dat`**（官方每 device 每天 10 段 20s 传输） |

**已生成 manifest（`data/`）**

| 文件 | 用途 |
|------|------|
| `manifest_days_iq1_day1_to_day5.csv` | Day1–4 train / Day5 val；120 行（24×5） |
| `manifest_cross_day_day1_day2.csv` | Day1→Day2 严格跨天 |
| `manifest_cross_day_day1_to_day5.csv` | 扩展跨天 |
| `manifest_all.csv` | 汇总（含 smoke / days 子集） |

**当前实验已用**：LODO Day1–5、Day1→Day2、Day1–4→Day5（均基于 IQ_1 + 24 类）。

### 1.2 Diff_Receivers

| 项 | 状态 |
|----|------|
| 路径 | `data/raw/osu_lora/Diff_Receivers_Setup/Diff_Receivers_Setup_Indoor_SameTx/` |
| 接收机 | RX1 + RX2 |
| 文件 | `RX{1,2}/Device{X}_IQ.dat`，各 25 个 IQ（含 raw Device9 文件，manifest 中排除） |
| 规模 | **50** 个 IQ 文件 |

**已生成 manifest**

| 文件 | split |
|------|-------|
| `manifest_rx1_to_rx2.csv` | RX1 train / RX2 val |
| `manifest_rx2_to_rx1.csv` | RX2 train / RX1 val |

**当前实验已用**：cross-receiver 直接迁移（80ep norm confirm、aug/CFO/CORAL sweep 等）。

### 1.3 其他已存在内容（非 IQ 数据）

`data/raw/osu_lora/` 下还有下载辅助物，**不算 setup 数据**：

- `aria2_*.txt`、`download_*.log`、`download_*.tsv`
- 本地 phase manifest：`manifest_smoke.csv`、`manifest_days_iq1.csv` 等

**本地不存在任何其他顶层 setup 目录**（Outdoor / Wired / Config / Location / Distance / 其他 Receivers 变体均未下载）。

---

## 2. 未下载 setup 清单

### 2.1 按顶层 setup

| Setup | 远程结构要点 | 预估 IQ 规模（最小切片） | 下载状态 |
|-------|-------------|-------------------------|----------|
| **Diff_Days Indoor** | `Day{d}/Device{n}/IQ_{1..10}.dat` | IQ_1 only: 125；全量: **1250** | 124/125 IQ_1 |
| **Diff_Days Outdoor** | 同 Indoor | IQ_1: **125**；全量: 1250 | 0 |
| **Diff_Days Wired** | `Day{d}/Device{n}/dev{n}_rx{1..10}_iq.dat` | 每 day 250 IQ；5 day ≈ **1250** | 0 |
| **Diff_Configurations** | `Config{c}/IQ_{1..25}.dat`（flat，`IQ_n` = Device n） | **100**（4×25） | 0 |
| **Diff_Locations** | `Location{l}/IQ_{1..25}.dat` | **75**（3×25） | 0 |
| **Diff_Distances** | `{5m,10m,15m,20m}/IQ_{1..25}.dat` | **100**（4×25） | 0 |
| **Diff_Receivers**（5 个未下变体） | 见下表 | 各 50 IQ；合计 **250** | 0 |

### 2.2 Diff_Receivers 六个子场景

| 子场景 | 接收机目录命名 | 文件命名 | 说明 |
|--------|---------------|----------|------|
| `Indoor_SameTx` | `RX1/`, `RX2/` | `Device{n}_IQ.dat` | **已下载** |
| `Indoor_DiffTx` | `Recv1/`, `Recv2/` | flat `IQ_{1..25}.dat`（+ FFT） | 不同发射位置，同 indoor |
| `Outdoor_SameTx` | `RX1/`, `RX2/` | `Device{n}_IQ.dat` | 室外同 TX 位置 |
| `Outdoor_DiffTx` | `RX1/`, `RX2/` | `Device{n}_IQ.dat` | 室外不同 TX 位置 |
| `Wired_SameTx` | `RX1/`, `RX2/` | `Device{n}_IQ.dat` | 有线同 TX |
| `Wired_DiffTx` | `RX1/`, `RX2/` | `Device{n}_IQ.dat` | 有线不同 TX |

---

## 3. 建议下载优先级

### P0 — Diff_Days complete / 当前已用（补齐即可）

| 任务 | 内容 | 理由 |
|------|------|------|
| P0a | 补 `Day2/Device9/IQ_1.dat`（若官方仍不可用则维持 exclude Device9） | 与现有 24 类 LODO 口径一致；retry 已在 `download_osu_lora_days345.sh` |
| P0b | （可选）Indoor Day1–5 的 **IQ_2..IQ_10** | 增加每类文件数，改善 file-level 统计稳定性；不改变 cross-day 协议 |
| P0c | 维持现有 IQ_1 manifest | **当前主实验线**，无需等新数据即可继续 |

> P0 **不阻塞** cross-config / cross-location 等新实验，但应优先于 P2 大规模 variant 下载。

### P1 — Diff_LoRa_Configuration

| 任务 | 内容 |
|------|------|
| 下载 | `Diff_Configurations_Setup/Config{1..4}/IQ_{1..25}.dat` |
| SF 映射 | Config1=SF7, Config2=SF8, Config3=SF11, Config4=SF12（BW 125kHz, CR 4/5, TX 20dBm） |

**理由**：OSU 论文报告 **SF 变更时 IQ 模型退化最严重**之一；是 IoTJ 部署鲁棒性的核心对照轴。

### P1 — Diff_Location / Diff_Distance

| 任务 | 内容 |
|------|------|
| Location | `Location{1..3}/IQ_{1..25}.dat` |
| Distance | `{5m,10m,15m,20m}/IQ_{1..25}.dat` |
| 语义 | Location1≈room, Location2≈office, Location3≈outdoor（论文 Section II-B）；Distance 为固定 SF7 下 TX–RX 距离扫掠 |

**理由**：channel / 几何变化是 RFFI 第二大部署 shift；体量小（175 IQ），性价比高。

### P2 — Outdoor / Indoor variants / 其他 Receivers

| 任务 | 内容 | 理由 |
|------|------|------|
| Diff_Days Outdoor | Day1–5 × IQ_1（或全量） | 室外跨天泛化；与 Indoor LODO 对照 |
| Diff_Days Wired | Day1–5 × dev{n}_rx{k}_iq | 有线采集跨天；channel 更可控 |
| Receivers 其余 5 变体 | Indoor DiffTx, Outdoor Same/DiffTx, Wired Same/DiffTx | 扩展 cross-receiver；DiffTx 分离 TX 位置 vs RX 增益 |
| Diff_Days IQ 全量 | Indoor/Outdoor 各 1250 IQ | RF-MAE / 大规模 pretrain |

### P3 — 外部 RFFI 数据集（仅参考，非 OSU 下载）

| 数据集 / 论文 | 用途 | 注意 |
|--------------|------|------|
| Shen et al. spectrogram-CNN LoRa（INFOCOM 2021） | 时频表示、CFO 补偿思路 | 设备数/协议不同，**不可直接对比 OSU Day1–5** |
| Scalable channel-robust LoRa RFFI（TIFS 2022, 60 devices） | metric learning / 扩设备参考 | 分割与 OSU 不兼容 |
| Channel-robust contrastive / receiver-independent（WCNC/arXiv 2025） | receiver-independent 动机 | 高 reported acc 仅作方向参考 |
| DeepCRF / 其他 WiFi-BT RFFI（OSU 同实验室其他 release） | 方法迁移参考 | 调制与带宽不同 |

---

## 4. 每个 setup 的用途

图例：● 主用途　○ 次要 / 辅助

| Setup | cross-day | cross-config | cross-location | cross-receiver | RF-MAE pretrain | style aug 统计 |
|-------|:---------:|:------------:|:--------------:|:--------------:|:---------------:|:--------------:|
| Diff_Days Indoor（IQ_1，已用） | ● | | | | ○ | ○ |
| Diff_Days Indoor（IQ_1..10 全量） | ● | | | | ● | ○ |
| Diff_Days Outdoor | ● | | ○ | | ● | ○ |
| Diff_Days Wired | ● | | | | ● | |
| Diff_Configurations | | ● | | | ● | ○ |
| Diff_Locations | | | ● | | ● | ● |
| Diff_Distances | | | ●（距离轴） | | ● | ● |
| Receivers Indoor SameTx（已用） | | | | ● | ○ | ● |
| Receivers Indoor DiffTx | | | | ● | ○ | ● |
| Receivers Outdoor * | ○ | | ○ | ● | ● | ● |
| Receivers Wired * | | | | ● | ○ | ● |

**用途说明**

- **cross-day**：LODO / Day1→Day2 / Day1–4→Day5；当前主结果来源。
- **cross-config**：Config1 train → Config2/3/4 test（或 leave-one-config-out）；验证 chirp/SF 变化下 OOB-attention 是否仍有效。
- **cross-location / cross-distance**：Location1→2/3 或 5m→10/15/20m；评估信道/几何 shift。
- **cross-receiver**：RX1↔RX2 直接迁移；Indoor SameTx 已证明 RX 增益污染 OOB，DiffTx/Outdoor/Wired 用于分离因素。
- **RF-MAE pretraining**：多 setup、多文件无标签 IQ 预训练 encoder；优先 Indoor Days 全量 + Config + Location。
- **style augmentation statistics**：估计 `--augment-receiver-style` 的 gain/noise/tilt/oob 分布；已有 `analyze_receiver_spectrum_stats.py` 可扩展到 Location/Distance/Outdoor。

---

## 5. 每个 setup 预期 manifest 规范

### 5.1 全局字段约定

与现有 manifest 保持一致（见 `generate_manifest_days.py` / `generate_manifest_receivers.py`）：

```text
path, relative_path, device, label, day, receiver, location, distance, sf, scene, config, setup, split
```

| 字段 | 规则 |
|------|------|
| `device` | 1..24（raw Device9 跳过；raw Device10→device 9，…，raw Device25→device 24） |
| `label` | **0..23**（= device − 1） |
| 排除 raw Device9 | **是**（所有 setup 统一） |
| 文件存在性 | manifest 生成前应用 `check_manifest.py` 校验 |

### 5.2 分 setup 预期 manifest

#### A. Diff_Days Indoor — 当前已用（IQ_1）

| 项 | 值 |
|----|-----|
| 建议输出 | `data/manifest_days_iq1_day1_to_day5.csv`（已有） |
| train | Day1, Day2, Day3, Day4（各 24 文件） |
| val/test | Day5（24 文件）；或 LODO 时 held-out day = val |
| device 映射 | raw Device{n} → device = n − (1 if n>9 else 0) |
| 排除 Device9 | 是 |
| label | 0..23 |
| setup 字段 | `diff_days_indoor` |

**LODO 变体**：5 个 manifest 或使用同一 CSV + 运行时按 `day` 过滤；每 fold 24 train-day 文件 × 4 days + 24 test-day 文件。

#### B. Diff_Days Indoor — 全量（IQ_1..10，P0b）

| 项 | 值 |
|----|-----|
| 建议输出 | `data/manifest_days_indoor_iq1to10.csv` |
| train/val | 与 IQ_1 相同 day split；每 (day, device) 10 行 |
| 规模 | 最多 5×24×10 = **1200** 行（仍排除 Device9；Day2 缺 Device9 则 1190） |
| setup 字段 | `diff_days_indoor_full` |

#### C. Diff_Configurations（P1）

| 项 | 值 |
|----|-----|
| 建议输出 | `data/manifest_config{1}_to_config{2,3,4}.csv` 或 LOCO 5-fold |
| 路径模板 | `Diff_Configurations_Setup/Config{c}/IQ_{raw_device}.dat` |
| 默认 split | train=Config1 (SF7)；val/test=Config2/3/4 各 24 文件 |
| `config` 字段 | 1..4；`sf` 字段 | 7, 8, 11, 12 |
| `scene` | `indoor` |
| setup 字段 | `diff_configurations` |
| 排除 Device9 | 是（跳过 `IQ_9.dat`） |
| label | 0..23 |

#### D. Diff_Locations（P1）

| 项 | 值 |
|----|-----|
| 建议输出 | `data/manifest_loc1_to_loc{2,3}.csv` |
| 路径模板 | `Diff_Locations_Setup/Location{l}/IQ_{raw_device}.dat` |
| 默认 split | train=Location1；val/test=Location2, Location3 |
| `location` 字段 | 1=room, 2=office, 3=outdoor |
| setup 字段 | `diff_locations` |
| 排除 Device9 | 是 |
| label | 0..23 |

#### E. Diff_Distances（P1）

| 项 | 值 |
|----|-----|
| 建议输出 | `data/manifest_dist5m_to_dist{10,15,20}m.csv` |
| 路径模板 | `Diff_Distances_Setup/{5m,10m,15m,20m}/IQ_{raw_device}.dat` |
| 默认 split | train=5m；val/test=10m, 15m, 20m |
| `distance` 字段 | 5, 10, 15, 20 |
| setup 字段 | `diff_distances` |
| 排除 Device9 | 是 |
| label | 0..23 |

#### F. Diff_Receivers — Indoor SameTx（已用）

| 项 | 值 |
|----|-----|
| 输出 | `manifest_rx1_to_rx2.csv`, `manifest_rx2_to_rx1.csv`（已有） |
| train/val | RX1→RX2 或 RX2→RX1；各 24 train + 24 val |
| `receiver` | 1=RX1, 2=RX2 |
| setup 字段 | `diff_receivers_indoor_sametx` |
| 排除 Device9 | 是 |
| label | 0..23 |

#### G. Diff_Receivers — 其他变体（P2）

| 子场景 | 路径模板 | receiver 命名 | setup 字段 |
|--------|----------|---------------|------------|
| Indoor DiffTx | `.../Indoor_DiffTx/Recv{r}/IQ_{n}.dat` | Recv1→1, Recv2→2 | `diff_receivers_indoor_difftx` |
| Outdoor SameTx | `.../Outdoor_SameTx/RX{r}/Device{n}_IQ.dat` | RX1/2 | `diff_receivers_outdoor_sametx` |
| Outdoor DiffTx | 同上 DiffTx | RX1/2 | `diff_receivers_outdoor_difftx` |
| Wired SameTx | `.../Wired_SameTx/RX{r}/Device{n}_IQ.dat` | RX1/2 | `diff_receivers_wired_sametx` |
| Wired DiffTx | 同上 | RX1/2 | `diff_receivers_wired_difftx` |

每个变体各生成双向 manifest（24 train + 24 val）；**排除 Device9**；**label 0..23**。

#### H. Diff_Days Outdoor / Wired（P2）

| Setup | 路径模板 | split 建议 | setup 字段 |
|-------|----------|------------|------------|
| Outdoor | 同 Indoor：`Day{d}/Device{n}/IQ_{k}.dat` | 与 Indoor LODO 平行（IQ_1 先行） | `diff_days_outdoor` |
| Wired | `Day{d}/Device{n}/dev{n}_rx{k}_iq.dat` | Day1–4 train / Day5 val；或 LODO | `diff_days_wired` |

Wired manifest 需新增 `receiver` 或 `rx_index` 解析规则（k=1..10）；**Device9 仍排除**。

#### I. RF-MAE pretrain manifest（跨 setup）

| 项 | 值 |
|----|-----|
| 建议输出 | `data/manifest_pretrain_osu_lora.csv` |
| split | 全部 `split=pretrain`（或按 setup 9:1 留 val 监控） |
| 标签 | 保留 device/label 供 optional supervised head；MAE 主任务不用 label |
| 优先级来源 | Indoor Days IQ_1..10 → Config → Location → Distance → Receivers |

---

## 6. 下载脚本缺口（规划，不执行）

| 优先级 | 建议脚本 | 状态 |
|--------|----------|------|
| P0 | `scripts/download_osu_lora_days345.sh` | **已有**（补 Day2/D9 + Day3–5 IQ_1） |
| P0b | `scripts/download_osu_lora_days_iq2to10.sh` | 待写 |
| P1 | `scripts/download_osu_lora_configs.sh` | 待写 |
| P1 | `scripts/download_osu_lora_locations_distances.sh` | 待写 |
| P2 | `scripts/download_osu_lora_days_outdoor.sh` | 待写 |
| P2 | `scripts/download_osu_lora_receivers_variants.sh` | 待写 |
| — | `scripts/generate_manifest_configs.py` 等 | 待写（可仿 `generate_manifest_days.py`） |

官方 base URL：

```text
https://research.engr.oregonstate.edu/hamdaoui/RFFP-dataset/LoRa-Dataset/
```

---

## 7. 推荐执行顺序（仍不自动下载）

```text
1. [P0] 确认 Day2/Device9 官方是否可补；维持 24 类 manifest
2. [P1] Diff_Configurations（100 IQ）→ 生成 cross-config manifest → 实验
3. [P1] Diff_Locations + Diff_Distances（175 IQ）→ cross-location manifest
4. [P0b] Indoor IQ_2..10（若 file-level 统计 / MAE 需要）
5. [P2] Outdoor Days、Receivers 变体
6. [P3] 外部数据集仅引用，不纳入 OSU 统一 manifest
```

---

## 8. 与当前实验的关系

| 已完成实验 | 依赖数据 | 扩展后可新增 |
|-----------|----------|-------------|
| LODO Day1–5 | Diff_Days Indoor IQ_1 | IQ 全量、Outdoor/Wired LODO |
| Cross-receiver RX1↔RX2 | Indoor SameTx | DiffTx / Outdoor / Wired variants |
| Norm / aug / CORAL sweep | 同上 | Location/Config shift 上验证 oob_ratio 是否仍最优 |
| Query Bottleneck（设计稿） | 不依赖新数据 | Config/Location 作第二验证轴 |

**本文档仅作数据盘点与优先级规划；不包含任何下载或训练命令。**
