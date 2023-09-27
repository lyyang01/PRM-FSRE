python train_demo.py \
    --trainN 5 --N 5 --K 1 --Q 1 --dot \
    --model proto --encoder bert --hidden_size 768 --val_step 1000 \
	--pretrain_ckpt path to bert-base-uncased \
    --batch_size 4 \
    --visualize \
    --cat_entity_rep \
    --load_ckpt ./checkpoint/5-1.pth.tar \