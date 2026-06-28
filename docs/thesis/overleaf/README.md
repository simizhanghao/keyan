# Overleaf 使用说明

## 上传方式

**方式 A：上传 ZIP（推荐）**

1. 下载或解压 `thesis_overleaf_pack.zip`
2. 打开 [Overleaf](https://www.overleaf.com) → New Project → Upload Project
3. 选择 zip 文件上传

**方式 B：Git 同步**

本目录在仓库 `lora-rffi-thesis/overleaf/` 下，可将 Overleaf 项目链接到 GitHub 仓库同步。

## 编译设置

| 项 | 值 |
|----|-----|
| Compiler | **XeLaTeX** |
| Main document | `main.tex` |
| 编译次数 | 2 遍（生成目录与交叉引用） |

Overleaf：Menu → Settings → Compiler → XeLaTeX

## 目录结构

```text
main.tex                 # 主文件
frontmatter/             # 中英文摘要
chapters/                # 第 1--6 章
tables/                  # 表格
figures/ch3|ch4|ch5/     # 已嵌入 PDF 图（17 张）
refs.bib                 # 参考文献（待扩充）
```

## 与 Markdown 写作包关系

| 格式 | 路径 |
|------|------|
| Markdown 草稿 | 仓库根目录 `chapters/*.md` |
| LaTeX Overleaf | 本目录 `chapters/*.tex` |

内容一致，LaTeX 版含图表引用与学校封面占位。

## 待人工完成

1. **封面**：按北航硕士模板替换 `main.tex` 中 `titlepage`
2. **参考文献**：补充 `refs.bib` 并在正文添加 `\cite`
3. **第 2 章** Table 2-1 数据集统计（待从 manifest 生成）
4. **页眉页脚**：按学院格式添加 `fancyhdr` 等
5. **查重前**：核对与 Paper 1/2 重复段落，毕设应整合叙述而非整段复制英文论文

## 实验与数值来源

- 代码：`https://github.com/hanCChan/rf-hstu-lora` 分支 `thesis-em-openset`
- Commit：`f9dbe1c`（第 5 章实验）、`b4f50f1`（写作包）

## 常见问题

**Q: 编译报错 fontset？**  
Overleaf 上 `ctexbook` 使用 `fontset=fandol`（已在 main.tex 设置）。

**Q: 找不到图片？**  
确认 `figures/ch3|ch4|ch5/` 下 PDF 已上传；勿只上传 tex 不含 figures。

**Q: 参考文献空白？**  
正文暂未大量 `\cite`；可改用 `biblatex` 或手动 `\begin{thebibliography}`。
