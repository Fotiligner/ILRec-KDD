export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

DATASET=Games
DATA_PATH=./data

# ckpts=("129" "258" "387" "514")
ckpts=("1177")

RESULTS_FILE=./results/$DATASET/xxx.json

for ckpt in "${ckpts[@]}"
do
    echo $ckpt
    CKPT_PATH=""
    echo $CKPT_PATH
    torchrun --nproc_per_node=8 --master_port=22500 test_ddp.py \
        --ckpt_path $CKPT_PATH \
        --dataset $DATASET \
        --data_path $DATA_PATH \
        --results_file $RESULTS_FILE \
        --test_batch_size 1 \
        --test_prompt_ids 0 \
        --num_beams 20 \
        --test_task seqrec \
        --index_file .index.json
done

