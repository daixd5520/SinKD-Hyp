# Sinkhorn Distance Minimization for Knowledge Distillation (COLING 2024 and TNNLS 2024)
## Installation
To install the environment, run:

`sh ins.sh`

## Download GLUE and SuperGLUE Data
Download the GLUE data using [this repository](https://github.com/nyu-mll/GLUE-baselines) or from GLUE benchmark website, unpack it to directory datas/glue and rename the folder `CoLA` to `COLA`.

Download the SuperGLUE data from SuperGLUE benchmark website.

## Download Pre-trained BERT
Download `bert_uncased_L-12_H-768_A-12` (BERT-base) and `bert_uncased_L-6_H-768_A-12` for teacher model and student model, respectively, from [this repository](https://github.com/google-research/bert). and use the [API from Huggingface](https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/convert_bert_original_tf_checkpoint_to_pytorch.py) to transform them to pytorch checkpoint.

## Task-specific BERT Model Distillation
The training script for **Task-specific Teacher Model Finetuning** can be found in the `script/teacher/` directory, where **$TEACHER_PATH** denotes the file path of the teacher model.

Similarly, the training script for **Task-specific Student Model Distillation** is located in the `script/student/` directory. In this case, **$STUDENT_PATH** and **$TEACHER_PATH** represent the file paths of the student and teacher models, respectively.

## 如何运行（以 GLUE 任务为例）
1. 安装依赖并准备数据与模型：
   - `sh ins.sh`
   - 将 GLUE 数据放到 `datas/glue` 下（如 `datas/glue/COLA` 等），并准备好教师/学生 BERT 权重目录。

2. 先微调教师模型（示例以 CoLA 为例，可直接调用 `main_glue.py`，`script/teacher/cola.sh` 也提供了默认超参的参考写法；此阶段仅使用常规交叉熵/微调损失，不需要双曲映射或 Sinkhorn）：
   ```bash
   CUDA_VISIBLE_DEVICES=0 python3 main_glue.py \
     --do_train --do_eval --do_lower_case \
     --task_name cola \
     --model_path <TEACHER_INIT_CKPT> \
     --data_dir ./datas/glue \
     --per_gpu_batch_size 2 \
     --num_train_epochs 8 \
     --learning_rate 2e-5
   ```

3. 使用 Sinkhorn 蒸馏学生模型（默认开启洛伦兹双曲距离，参见 `loss.py` 中 `Sinkhorn` 类）：
   ```bash
   CUDA_VISIBLE_DEVICES=0 python3 main_glue_distill.py \
     --do_train --do_eval \
     --task_name cola \
     --teacher_path <FINETUNED_TEACHER_DIR> \
     --student_path <STUDENT_INIT_DIR> \
     --data_dir ./datas/glue \
     --distill_loss kd+sinkhorn \
     --alpha 0.9 --beta 0.8 --temperature 1.0 \
     --per_gpu_batch_size 16 --num_train_epochs 20
   ```
   `--distill_loss kd+sinkhorn` 会在交叉熵与 KD 之外加入双曲 Sinkhorn 损失；如需仅使用 Sinkhorn，可改为 `--distill_loss sinkhorn`。曲率是否可学习由 `loss.py` 中 `Sinkhorn` 的构造参数控制，当前脚本默认使用固定曲率。

## Task-specific T0 Model Distillation
To install the environment, run:

`sh T0/ins.sh`

To perform **Task-specific Teacher Model Finetuning**, run:

`python3 T0/distillation_t.py     --dataset_name super_glue     --dataset_config_name DATASET_NAME     --template_name "TEMPLATE_NAME"     --model_name_or_path MODEL_DIR     --output_dir ./debug    --parallelize `

To perform **Task-specific Student Model Distillation**, run:

`python3 T0/distillation.py     --dataset_name super_glue     --dataset_config_name DATASET_NAME     --template_name "TEMPLATE_NAME"     --model_name_or_path MODEL_DIR     --output_dir ./debug    --parallelize `

## Task-specific GPT Model Distillation
To install the environment, run:

`sh GPT-Neo/ins.sh`

To perform **Task-specific Teacher Model Finetuning**, run:

`python3 GPT-Neo/distillation_t.py     --dataset_name super_glue     --dataset_config_name DATASET_NAME     --template_name "TEMPLATE_NAME"     --model_name_or_path MODEL_DIR     --output_dir ./debug    --parallelize `

To perform **Task-specific Student Model Distillation**, run:

`python3 GPT-Neo/distillation.py     --dataset_name super_glue     --dataset_config_name DATASET_NAME     --template_name "TEMPLATE_NAME"     --model_name_or_path MODEL_DIR     --output_dir ./debug    --parallelize `

## Student Checkpoints
The distilled student model for each task reported in the paper can be downloaded using the following link:
https://drive.google.com/drive/folders/1BsA0VHKSa_-Bp5I7dQ2Ftk2q7cIyPrdC

## Teacher Checkpoints
The teacher model for each task reported in the paper can be downloaded using the following link:
https://drive.google.com/file/d/1sBi35Dk8VJ7TU0warB6BL9QKx-in9Ww6/view?usp=drive_link

## BibTeX
```
@article{cui2024sinkhorn,
  title={Sinkhorn Distance Minimization for Knowledge Distillation},
  author={Cui, Xiao and Qin, Yulei and Gao, Yuting and Zhang, Enwei and Xu, Zihan and Wu, Tong and Li, Ke and Sun, Xing and Zhou, Wengang and Li, Houqiang},
  journal={arXiv preprint arXiv:2402.17110},
  year={2024}
}

@article{cui2024sinkhorn2,
  title={SinKD: Sinkhorn Distance Minimization for Knowledge Distillation},
  author={Cui, Xiao and Qin, Yulei and Gao, Yuting and Zhang, Enwei and Xu, Zihan and Wu, Tong and Li, Ke and Sun, Xing and Zhou, Wengang and Li, Houqiang},
  journal={TNNLS},
  year={2024},
  publisher={IEEE}
}
```

