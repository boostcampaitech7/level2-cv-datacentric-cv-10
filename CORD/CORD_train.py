import os
import os.path as osp
import time
import math
from datetime import timedelta
from argparse import ArgumentParser

import torch
from torch import cuda
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from tqdm import tqdm
import wandb
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from east_dataset import EASTDataset
from model import EAST
from loss import EASTLoss
from dataset import SceneTextDataset  # SceneTextDataset 클래스 불러오기
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def parse_args():
    parser = ArgumentParser()

    # Conventional args
    parser.add_argument('--data_dir', type=str, default='/data/CORD')
    parser.add_argument('--model_dir', type=str, default='trained_models')
    parser.add_argument('--device', default='cuda' if cuda.is_available() else 'cpu')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--image_size', type=int, default=2048)
    parser.add_argument('--input_size', type=int, default=1024)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--max_epoch', type=int, default=150)
    parser.add_argument('--save_interval', type=int, default=1)
    
    args = parser.parse_args()
    if args.input_size % 32 != 0:
        raise ValueError('`input_size` must be a multiple of 32')

    return args

def load_dataset(root_dir, split, image_size, input_size, is_train=False):
    if is_train == True:
        augmentation = A.Compose([
        A.ShiftScaleRotate(shift_limit=0.02, scale_limit=0.1, rotate_limit=10, border_mode=cv2.BORDER_CONSTANT, p=0.7),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
        A.MotionBlur(blur_limit=3, p=0.2),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=5, val_shift_limit=5, p=0.3),
        A.Normalize(mean=(0.5,), std=(0.5,))
    ], bbox_params=A.BboxParams(format='polygon', label_fields=['labels']))
    elif is_train == False:
        augmentation = False

    dataset = SceneTextDataset(
        root_dir=root_dir,
        split=split,
        image_size=image_size,
        crop_size=input_size,
        transform=augmentation
    )
    dataset = EASTDataset(dataset)
    return dataset

def do_training(data_dir, model_dir, device, image_size, input_size, num_workers, batch_size,
                learning_rate, max_epoch, save_interval):
    # WandB 초기화
    wandb.init(entity="Entitiy Name", project="CORD_training", name="CORD_augset2_Detection")
    
    train_dataset = load_dataset(data_dir, 'train', image_size, input_size, is_train=True)
    val_dataset = load_dataset(data_dir, 'dev', image_size, input_size, is_train=False)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = EAST().to(device)
    criterion = EASTLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[max_epoch // 2], gamma=0.1)

    best_iou_loss = float('inf')  # Initialize best IOU loss to a high value
    recent_checkpoints = []  # 최근 저장된 체크포인트 파일의 경로를 저장할 리스트

    for epoch in range(max_epoch):
        model.train()
        epoch_loss, epoch_start = 0, time.time()

        with tqdm(total=len(train_loader)) as pbar:
            for img, gt_score_map, gt_geo_map, roi_mask in train_loader:
                img, gt_score_map, gt_geo_map, roi_mask = img.to(device), gt_score_map.to(device), gt_geo_map.to(device), roi_mask.to(device)
                
                optimizer.zero_grad()
                pred_score, pred_geo = model(img)
                loss, loss_details = criterion(gt_score_map, pred_score, gt_geo_map, pred_geo, roi_mask)
                
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                pbar.update(1)
                # 학습 손실 로그 기록
                pbar.set_postfix(loss=loss.item(), cls_loss=loss_details['cls_loss'], angle_loss=loss_details['angle_loss'], iou_loss=loss_details['iou_loss'])

                # # 각 배치 손실 출력
                # print(f"Epoch [{epoch+1}/{max_epoch}], Batch Loss: {loss.item()}, Cls Loss: {loss_details['cls_loss']}, Angle Loss: {loss_details['angle_loss']}, IoU Loss: {loss_details['iou_loss']}")
                

                # WandB 학습 로그 기록
                wandb.log({
                    "train_loss": loss.item(),
                    "train_cls_loss": loss_details["cls_loss"],
                    "train_angle_loss": loss_details["angle_loss"],
                    "train_iou_loss": loss_details["iou_loss"]})
                
        # Validation
        model.eval()
        val_loss, val_cls_loss, val_angle_loss, val_iou_loss = 0, 0, 0, 0
        with torch.no_grad():
            for img, gt_score_map, gt_geo_map, roi_mask in val_loader:
                img, gt_score_map, gt_geo_map, roi_mask = img.to(device), gt_score_map.to(device), gt_geo_map.to(device), roi_mask.to(device)
                pred_score, pred_geo = model(img)
                loss, loss_details = criterion(gt_score_map, pred_score, gt_geo_map, pred_geo, roi_mask)

                val_loss += loss.item()
                val_cls_loss += loss_details["cls_loss"]
                val_angle_loss += loss_details["angle_loss"]
                val_iou_loss += loss_details["iou_loss"]

        # Calculate average validation losses
        avg_val_loss = val_loss / len(val_loader)
        avg_val_cls_loss = val_cls_loss / len(val_loader)
        avg_val_angle_loss = val_angle_loss / len(val_loader)
        avg_val_iou_loss = val_iou_loss / len(val_loader)
        # 각 에폭의 검증 손실 출력
        print(f"Epoch [{epoch+1}/{max_epoch}], Train Loss: {epoch_loss/len(train_loader)}, Val Loss: {avg_val_loss}, Val Cls Loss: {avg_val_cls_loss}, Val Angle Loss: {avg_val_angle_loss}, Val IoU Loss: {avg_val_iou_loss}")

        # 평균 검증 손실 로그 기록 및 WandB에 기록
        wandb.log({
            "epoch": epoch + 1,
            "avg_train_loss": epoch_loss / len(train_loader),
            "avg_val_loss": avg_val_loss,
            "val_cls_loss": avg_val_cls_loss,
            "val_angle_loss": avg_val_angle_loss,
            "val_iou_loss": avg_val_iou_loss,
            "learning_rate": optimizer.param_groups[0]['lr']
        })

        # Save model if current epoch has the lowest IOU loss
        if avg_val_iou_loss < best_iou_loss:
            best_iou_loss = avg_val_iou_loss
            best_ckpt_fpath = osp.join(model_dir, 'best_model_iou.pth')
            torch.save(model.state_dict(), best_ckpt_fpath)
            print(f"Saved best model with IOU loss: {best_iou_loss:.4f}")

        scheduler.step()

        # 주기적으로 모델 저장
        if (epoch + 1) % save_interval == 0:
            if not osp.exists(model_dir):
                os.makedirs(model_dir)
            
            ckpt_fpath = osp.join(model_dir, f"east_model_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), ckpt_fpath)
            print(f"Saved checkpoint: {ckpt_fpath}")
            
            # 최근 저장된 모델 경로 리스트에 추가
            recent_checkpoints.append(ckpt_fpath)

            # 최근 모델이 3개를 초과할 경우 가장 오래된 모델 삭제
            if len(recent_checkpoints) > 3:
                oldest_ckpt = recent_checkpoints.pop(0)
                if osp.exists(oldest_ckpt):
                    os.remove(oldest_ckpt)
                    print(f"Removed old checkpoint: {oldest_ckpt}")

        print("Training completed.")
        wandb.finish()

def main(args):
    do_training(**vars(args))

if __name__ == '__main__':
    args = parse_args()
    main(args)
