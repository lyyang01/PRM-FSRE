python train_demo.py \
    --trainN 10 --N 10 --K 5 --Q 1 --dot \
    --model proto --encoder bert --hidden_size 768 --val_step 1000 \
    --pretrain_ckpt path to bert-base-uncased \
    --test_iter 1000 \
    --batch_size 4 \
    --cat_entity_rep --only_test \
    --load_ckpt ./checkpoint/seed/10-5-seed-43.pth.tar \
#    --is_pubmed \