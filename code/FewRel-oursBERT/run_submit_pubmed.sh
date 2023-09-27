python train_demo.py \
    --trainN 10 --N 10 --K 5 --Q 1 --dot \
    --model proto --encoder bert --hidden_size 768 --val_step 1000 --test pubmed-10-5 \
    --batch_size 4 --test_online --load_ckpt ./checkpoint/pubmed/10-5-pubmed.pth.tar \
    --pretrain_ckpt path to bert-base-uncased \
    --test_output ./submit/pubmed/pred-10-5.json \
    --cat_entity_rep \
    --test_iter 2500 \
    --is_pubmed \