cd src
# train
#python main.py ctdet --exp_id coco_dla_bill_res18 --dataset coco_bill --batch_size 32 --master_batch 14 --lr 1.25e-4  --gpus 0,1 --arch res_18  --save_all --resume
python main.py ctdet --exp_id coco_dla_bill_res18_2 --dataset coco_bill --batch_size 50 --master_batch 12 --lr 1.25e-4  --gpus 0,1 --arch res_18  --save_all 
#python main.py ctdet --exp_id coco_dla_1x --batch_size 128 --master_batch 9 --lr 5e-4 --gpus 0,1,2,3,4,5,6,7 --num_workers 16
# test
#python test.py ctdet --exp_id coco_dla_1x --keep_res --resume
# flip test
#python test.py ctdet --exp_id coco_dla_1x --keep_res --resume --flip_test 
# multi scale test
#python test.py ctdet --exp_id coco_dla_1x --keep_res --resume --flip_test --test_scales 0.5,0.75,1,1.25,1.5
cd ..
