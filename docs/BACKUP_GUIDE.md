# 毕设全量备份与快速下载指南

## 重要结论

| 内容 | 体积 | GitHub 能否存 | 推荐方式 |
|------|------|---------------|----------|
| 代码 + 脚本 + 文档 | ~100 MB | ✅ 可以 | `rf-hstu-lora` 仓库 |
| 毕设 LaTeX + 图表 + PDF | ~50 MB | ✅ 可以 | `lora-rffi-thesis` 仓库 + Release |
| 冻结实验 CSV/图/checkpoint | ~20 MB | ✅ 可以 | 同上（已在代码仓） |
| **原始 IQ 数据 `data/raw/`** | **67 GB** | ❌ 不行 | 本地 tar / 网盘 / 第二块硬盘 |

GitHub 单文件上限 **100 MB**，免费仓建议 **< 1 GB**；67 GB 原始数据请用下方「本地打包」或网盘。

GitHub 公开仓库**长期保留**（无 3 年到期限制），适合代码与论文；原始大数据请另做一份物理备份。

---

## 一、GitHub 一键克隆（代码 + 毕设 + 冻结结果，不含 raw）

```bash
# 实验代码仓（毕设分支，含第 3–5 章冻结结果）
git clone --branch thesis-em-openset --depth 1 \
  https://github.com/hanCChan/rf-hstu-lora.git

# 毕设写作仓（LaTeX + Markdown + 图表）
git clone --depth 1 \
  https://github.com/hanCChan/lora-rffi-thesis.git

# 下载毕设 PDF（GitHub Release，无需 git）
gh release download -R hanCChan/lora-rffi-thesis latest
# 或浏览器打开：
# https://github.com/hanCChan/lora-rffi-thesis/releases/latest
```

**关键链接：**

- 代码：https://github.com/hanCChan/rf-hstu-lora/tree/thesis-em-openset
- 毕设：https://github.com/hanCChan/lora-rffi-thesis
- Overleaf 包：https://github.com/hanCChan/lora-rffi-thesis/blob/master/thesis_overleaf_pack.zip

---

## 二、本地全量打包（含 67 GB 原始数据）

在服务器 `/data1/hcc` 执行：

```bash
bash llm4RF/scripts/backup_full_project.sh
```

输出目录默认：`/data1/hcc/backups/`，生成三个包：

| 文件 | 内容 | 约体积 |
|------|------|--------|
| `thesis_writing_YYYYMMDD.tar.gz` | lora-rffi-thesis 全仓 | ~30 MB |
| `code_results_YYYYMMDD.tar.gz` | llm4RF 代码+结果（**不含** data/raw） | ~80 MB |
| `raw_osu_lora_YYYYMMDD.tar.gz` | 原始 IQ（仅 data/raw/osu_lora） | ~67 GB |

### 下载到个人电脑

**方式 A — rsync（推荐，可断点续传）**

```bash
# 在你自己电脑上执行（把 USER@SERVER 换成你的 SSH）
rsync -avP --partial USER@SERVER:/data1/hcc/backups/ ./thesis_backup/
```

**方式 B — scp 单文件**

```bash
scp USER@SERVER:/data1/hcc/backups/code_results_*.tar.gz .
scp USER@SERVER:/data1/hcc/backups/thesis_writing_*.tar.gz .
# 原始数据 67GB，建议 rsync 或硬盘拷贝，不要用 scp 一次性传
```

**方式 C — 外接硬盘**

在服务器上直接 `cp /data1/hcc/backups/*.tar.gz /mnt/usb/`

---

## 三、原始数据的其他长期保存选项

1. **百度网盘 / 阿里云盘** — 上传 `raw_osu_lora_*.tar.gz`（国内下载快）
2. **Zenodo** — 学术存档，单 deposit 最大 50 GB，可拆成 2 个 deposit
3. **Hugging Face Dataset** — 公开数据集免费托管（需按 HF 格式整理）
4. **学校 NAS / 实验室存储** — 问导师是否有长期存储配额

---

## 四、恢复环境

```bash
# 1. 解压代码包或 git clone
tar -xzf code_results_YYYYMMDD.tar.gz
cd rf-hstu-lora   # 或 llm4RF 目录名

# 2. 解压原始数据到 data/raw/
tar -xzf raw_osu_lora_YYYYMMDD.tar.gz -C /path/to/rf-hstu-lora/data/raw/

# 3. 毕设 PDF
tar -xzf thesis_writing_YYYYMMDD.tar.gz
# 或 gh release download -R hanCChan/lora-rffi-thesis latest
```

---

## 五、建议的「三年备份」组合

| 层级 | 存什么 | 放哪里 |
|------|--------|--------|
| L1 必做 | 两仓 GitHub + Release PDF | 永久公开/私有仓 |
| L2 推荐 | `code_results` + `thesis_writing` tar | 个人电脑 + 网盘 |
| L3 必做（唯一 raw 副本时） | `raw_osu_lora` tar | 外接硬盘 或 网盘 |

每 6 个月：`git pull` 确认 GitHub 最新，重新跑 `backup_full_project.sh` 更新 tar 时间戳。
