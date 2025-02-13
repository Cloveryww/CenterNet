cd src
# train
#python main.py ctdet --exp_id coco_dla_bill --dataset coco_bill --batch_size 32 --master_batch 14 --lr 1.25e-4  --gpus 0,1
#python main.py ctdet --exp_id coco_dla_1x --batch_size 128 --master_batch 9 --lr 5e-4 --gpus 0,1,2,3,4,5,6,7 --num_workers 16
# test
python test.py ctdet --exp_id coco_dla_bill4  --dataset coco_bill  --resume --arch dlav0_34 --load_model /home/xiejinluo/yww/CenterNet/exp/ctdet/coco_dla_bill4/model_best.pth
#python test.py ctdet --exp_id coco_dla_bill  --dataset coco_bill --keep_res --resume
# flip test
#python test.py ctdet --exp_id coco_dla_1x --keep_res --resume --flip_test 
# multi scale test
#python test.py ctdet --exp_id coco_dla_1x --keep_res --resume --flip_test --test_scales 0.5,0.75,1,1.25,1.5
cd ..
