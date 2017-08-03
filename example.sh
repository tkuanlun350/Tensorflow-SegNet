#!/bin/bash

# for training
python main.py --log_dir=/tmp3/first350/TensorFlow/Logs/ --image_dir=/tmp3/first350/SegNet-Tutorial/CamVid/train.txt --val_dir=/tmp3/first350/SegNet-Tutorial/CamVid/val.txt --batch_size=5

# for finetune from saved ckpt
# python main.py --finetune=/tmp3/first350/TensorFlow/Logs/model.ckpt-1000  --log_dir=/tmp3/first350/TensorFlow/Logs/ --image_dir=/tmp3/first350/SegNet-Tutorial/CamVid/train.txt --val_dir=/tmp3/first350/SegNet-Tutorial/CamVid/val.txt --batch_size=5

#for testing
# python main.py --testing=/tmp3/first350/TensorFlow/Logs/model.ckpt-19000  --log_dir=/tmp3/first350/TensorFlow/Logs/ --test_dir=/tmp3/first350/SegNet-Tutorial/CamVid/test.txt --batch_size=5 --save_image=True
