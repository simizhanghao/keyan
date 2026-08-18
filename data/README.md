# 数据目录说明

这个目录只放数据相关内容，不放模型代码。

建议结构：

```text
data/
  raw/
    osu_lora/
      OSU 官方原始 .dat / .sigmf-meta
      manifest.csv
  processed/
    后续可放切好的 window、缓存 embedding 或 token
```

当前第一版直接从 `.dat` 读取 IQ window，不强制预处理成小文件。
