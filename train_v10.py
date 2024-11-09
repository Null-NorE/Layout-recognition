import sys
import argparse
import os

# sys.path.append('/root/ultralyticsPro/') # Path 以Autodl为例

from ultralytics import YOLO

import torch

# 确认使用的设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"--------------Using device: {device}-----------------")

def main(opt):
    yaml = opt.cfg
    model = YOLO(yaml) 

    model.info()
    model.to(device)
    
    # 输出模型的设备
    print(f"Model is on device: {next(model.parameters()).device}")

    # 训练时的详细信息
    results = model.train(data='data.yaml',  # 训练参数均可以重新设置
                          epochs=300, 
                          imgsz=640, 
                          workers=16, 
                          batch=32,
                          verbose=True, # 打印每个 epoch 的详细信息
                          )
    
    # 可以在训练后进行进一步的分析
    print("Training complete!")
    print(results)  # 打印训练结果摘要

def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='ultralytics/cfg/models/v10/yolov10m.yaml', help='initial weights path')
    parser.add_argument('--weights', type=str, default='', help='')

    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)