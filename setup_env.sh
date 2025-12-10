#!/usr/bin/env bash
set -euo pipefail

# ===== 基本路径 =====
ROOT="${ROOT:-$PWD}"
DATA_DIR="$ROOT/datas"
GLUE_DIR="$DATA_DIR/glue"
SUPERGLUE_DIR="$DATA_DIR/superglue"
MODELS_PT="$ROOT/models/bert_pt"
mkdir -p "$GLUE_DIR" "$SUPERGLUE_DIR" "$MODELS_PT"

# ===== Python 依赖（走清华镜像）=====
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -U "torch>=2.1" "transformers>=4.40" "datasets>=2.14" huggingface_hub

# ===== 启用 Hugging Face 国内镜像 =====
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_ENABLE_HF_TRANSFER=1   # 打开并行加速

# ===== 1) GLUE（含 MRPC 兜底）=====
if [[ ! -d "$ROOT/GLUE-baselines" ]]; then
  git clone --depth=1 https://github.com/nyu-mll/GLUE-baselines "$ROOT/GLUE-baselines"
fi
pushd "$ROOT/GLUE-baselines" >/dev/null
# 先尝试直接下载（大多可直接下）
python download_glue_data.py --data_dir "$GLUE_DIR" --tasks all || true
# MRPC 常因 s3 被墙失败，按官方提示用 GitHub 兜底
if [[ ! -d "$GLUE_DIR/MRPC" ]]; then
  echo "[Info] MRPC 直连失败，使用官方兜底路径..."
  git clone --depth=1 https://github.com/wasiahmad/paraphrase_identification.git
  python download_glue_data.py --data_dir "$GLUE_DIR" --tasks all \
    --path_to_mrpc=paraphrase_identification/dataset/msr-paraphrase-corpus
fi
popd >/dev/null
# 规范化大小写：CoLA -> COLA
[ -d "$GLUE_DIR/CoLA" ] && mv "$GLUE_DIR/CoLA" "$GLUE_DIR/COLA"

# ===== 2) SuperGLUE（用 datasets 经 hf-mirror 下）=====
python - <<'PY'
import os
os.environ.setdefault("HF_ENDPOINT","https://hf-mirror.com")
from datasets import load_dataset
tasks=["boolq","cb","copa","multirc","record","rte","wic","wsc"]
outdir=os.path.abspath("datas/superglue")
os.makedirs(outdir, exist_ok=True)
for t in tasks:
    ds=load_dataset("super_glue", t)
    td=os.path.join(outdir,t); os.makedirs(td, exist_ok=True)
    for split in ds:
        # 导出为 jsonl（通用、稳妥）
        ds[split].to_json(os.path.join(td,f"{split}.jsonl"), lines=True, force_ascii=False)
print("[OK] SuperGLUE downloaded via hf-mirror")
PY

# ===== 3) BERT（直接拿国内镜像的 PyTorch 版，无需转换）=====
python - <<'PY'
from transformers import AutoModel, AutoTokenizer
import os
pairs=[
 ("bert-base-uncased", "teacher_bert_uncased_L-12_H-768_A-12"),
 ("google/bert_uncased_L-6_H-768_A-12", "student_bert_uncased_L-6_H-768_A-12"),
]
save_root=os.path.abspath("models/bert_pt"); os.makedirs(save_root, exist_ok=True)
for model_id, local in pairs:
    path=os.path.join(save_root, local)
    tok=AutoTokenizer.from_pretrained(model_id)
    mdl=AutoModel.from_pretrained(model_id)
    tok.save_pretrained(path); mdl.save_pretrained(path)
    print("[OK] {} -> {}".format(model_id, path))
PY

echo
echo "✅ 完成："
echo "  - GLUE       -> $GLUE_DIR   （已将 CoLA 重命名为 COLA）"
echo "  - SuperGLUE  -> $SUPERGLUE_DIR （jsonl 按任务/切分存放）"
echo "  - BERT(Pt)   -> $MODELS_PT/teacher_* 与 student_*"
