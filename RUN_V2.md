# Mini-IGN v2

先不要重新下载数据。

## 1. 当前诊断
在原项目根目录运行：

```bash
python /path/to/diagnose_current.py
```

## 2. 新训练
把 `train_v2.py` 复制到原项目 `src/` 下，然后运行：

```bash
python src/train_v2.py \
  --features outputs/features.json \
  --out_dir outputs
```

它会比较：

- training-fold mean baseline
- Ridge
- small MLP

并同时比较：

- ligand_only
- ligand_plus_pocket

主要修复：

1. target y 在每个 outer training fold 内标准化
2. MLP 缩小到 64 -> 32 -> 1
3. 用 inner validation 做 early stopping
4. 增加 Ridge baseline
5. mean baseline 只使用 training fold 的均值，避免 CV 中把测试标签泄漏给 baseline
