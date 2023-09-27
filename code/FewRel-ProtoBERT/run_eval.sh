CUDA_VISIBLE_DEVICES=0 python train_demo.py \
    --trainN 10 --N 10 --K 5 --Q 1 --dot \
    --model proto --encoder bert --hidden_size 768 --val_step 1000 --pretrain_ckpt /path/to/bert-base-uncased \
    --batch_size 4 --save_ckpt /data/liuyang/fewrel_model/final-10-5-my-temp.pth.tar \
    --cat_entity_rep --only_test --load_ckpt /path/to/model \