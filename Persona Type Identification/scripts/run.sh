#!/usr/bin bash

for s in 0 1 2 3 4 5 6 7 8 9
do 
    python -u ADB.py \
        --dataset meld \
        --seed $s \
        --freeze_bert_parameters \
        --bert_model BERT/uncased_L-12_H-768_A-12/ \
        --save_model \
        --train_batch_size 8 \
        --eval_batch_size 8
done