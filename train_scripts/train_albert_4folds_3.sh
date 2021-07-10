export CUR_FOLD=3
export NUM_GPUS=4
export CUDA_VISIBLE_DEVICES=0,1,5,7
export DATA_DIR=./squad_v2
export OUTPUT_DIR=./albert-xxlarge-v2/finetuned_ckpts_4folds_$CUR_FOLD
export MODEL_DIR=./albert-xxlarge-v2/pretrained_model
export DATALIST_FILE=./folds_txt/train_4folds_$CUR_FOLD.txt

python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS \
folds_scripts/run_squad_nfolds.py \
    --model_type albert \
    --model_name_or_path $MODEL_DIR \
    --data_dir $DATA_DIR \
    --do_train \
    --do_lower_case \
    --datalist_file $DATALIST_FILE \
    --predict_file dev-v2.0.json \
    --per_gpu_train_batch_size=4 \
    --per_gpu_eval_batch_size=32 \
    --learning_rate 1e-5 \
    --num_train_epochs 2 \
    --max_seq_length 384 \
    --doc_stride 128 \
    --output_dir $OUTPUT_DIR \
    --save_steps 2000 \
    --threads 4 \
    --version_2_with_negative \
    --overwrite_output_dir \
    --fp16
