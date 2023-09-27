python train_demo.py \
    --trainN 10 --N 10 --K 5 --Q 1 --dot \
    --model proto --encoder bert --hidden_size 768 --val_step 1000 --test 10-5-test-relid \
    --batch_size 4 --test_online --load_ckpt ./checkpoint/10-5-add.pth.tar \
    --pretrain_ckpt path to bert-base-uncased \
    --test_output ./submit/add/pred-10-5.json \
    --cat_entity_rep \
    --test_iter 2500