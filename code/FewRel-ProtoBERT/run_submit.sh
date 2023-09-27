CUDA_VISIBLE_DEVICES=7 python train_demo.py \
    --trainN 10 --N 10 --K 5 --Q 1 --dot \
    --model proto --encoder bert --hidden_size 768 --val_step 1000 --test 10-5-test-relid \
    --test_output ./submit/pred-10-5.json \
    --batch_size 4 --test_online --load_ckpt /path/to/model \
    --cat_entity_rep \