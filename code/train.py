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
import random

from east_dataset import EASTDataset
from dataset import SceneTextDataset
from sklearn.model_selection import KFold
from model import EAST
from torch.utils.data import Subset


def parse_args():
    parser = ArgumentParser()

    # Conventional args
    parser.add_argument('--data_dir', type=str,
                        default=os.environ.get('SM_CHANNEL_TRAIN', 'last_data'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR',
                                                                        'trained_models'))

    parser.add_argument('--device', default='cuda' if cuda.is_available() else 'cpu')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--image_size', type=int, default=2048)
    parser.add_argument('--input_size', type=int, default=1024)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=5e-3)
    parser.add_argument('--max_epoch', type=int, default=150)
    parser.add_argument('--save_interval', type=int, default=5)

    parser.add_argument('--wandb_project', type=str, default='ocr_project')

    parser.add_argument('--fold', type=int, default=4)  # 현재 fold 번호
    parser.add_argument('--n_folds', type=int, default=5)  # 총 fold 수

    args = parser.parse_args()

    if args.input_size % 32 != 0:
        raise ValueError('`input_size` must be a multiple of 32')

    return args

def create_fold_datasets(dataset, fold_idx, n_folds):
    """데이터셋을 9:1로 분할하되, fold마다 다른 검증 세트 사용"""
    # extractor: 기존 dataset, simple: 외부 dataset
    lang_data = {
        'chinese': {'extractor': [], 'simple': []},
        'japanese': {'extractor': [], 'simple': []},
        'thai': {'extractor': [], 'simple': []},
        'vietnamese': {'extractor': [], 'simple': []}
    }
    
    # 데이터 분류
    for idx in range(len(dataset)):
        fname = dataset.image_fnames[idx]
        
        # 언어 판별 및 분류 
        if 'zh' in fname:
            lang = 'chinese'
        elif 'ja' in fname:
            lang = 'japanese'
        elif 'th' in fname:
            lang = 'thai'
        elif 'vi' in fname:
            lang = 'vietnamese'
        else:
            continue
            
        if fname.startswith('extractor'):
            lang_data[lang]['extractor'].append(idx)
        else:
            lang_data[lang]['simple'].append(idx)
    
    train_indices = []
    val_indices = []
    stats = {
        'total': len(dataset),
        'train': {'total': 0, 'extractor': 0, 'simple': 0},
        'val': {'total': 0, 'extractor': 0, 'simple': 0},
        'languages': {}
    }
    
    # 각 언어별로 분할
    random.seed(42 + fold_idx)  # fold마다 다른 시드 사용
    
    for lang in lang_data:
        extractor_indices = lang_data[lang]['extractor']
        simple_indices = lang_data[lang]['simple']
        
        # 데이터 셔플
        random.shuffle(extractor_indices)
        random.shuffle(simple_indices)
        
        # 각 언어별 검증 셋 크기 계산 (90:10)
        lang_total = len(extractor_indices) + len(simple_indices)
        if lang_total == 0:
            continue
            
        val_size = int(lang_total * 0.1)  # 10%
        n_val_extractor = int(val_size * 0.7)  # 검증 셋의 70%를 extractor에서
        n_val_simple = val_size - n_val_extractor  # 나머지는 simple에서
        
        # fold_idx를 사용하여 다른 부분을 검증 세트로 선택
        start_idx_extractor = (fold_idx * n_val_extractor) % len(extractor_indices)
        start_idx_simple = (fold_idx * n_val_simple) % len(simple_indices)
        
        # 순환식으로 인덱스 선택
        val_extractor = extractor_indices[start_idx_extractor:start_idx_extractor + n_val_extractor]
        val_simple = simple_indices[start_idx_simple:start_idx_simple + n_val_simple]
        
        # 나머지를 학습 세트로 선택
        train_extractor = [idx for idx in extractor_indices if idx not in val_extractor]
        train_simple = [idx for idx in simple_indices if idx not in val_simple]
        
        # 인덱스 추가
        train_indices.extend(train_extractor + train_simple)
        val_indices.extend(val_extractor + val_simple)
        
        # 통계 수집 
        stats['languages'][lang] = {
            'train': {
                'total': len(train_extractor) + len(train_simple),
                'extractor': len(train_extractor),
                'simple': len(train_simple)
            },
            'val': {
                'total': len(val_extractor) + len(val_simple),
                'extractor': len(val_extractor),
                'simple': len(val_simple)
            }
        }
        
        # 전체 통계 업데이트
        stats['train']['total'] += len(train_extractor) + len(train_simple)
        stats['train']['extractor'] += len(train_extractor)
        stats['train']['simple'] += len(train_simple)
        stats['val']['total'] += len(val_extractor) + len(val_simple)
        stats['val']['extractor'] += len(val_extractor)
        stats['val']['simple'] += len(val_simple)
    
    # 최종 셔플
    random.shuffle(train_indices)
    random.shuffle(val_indices)
    
    return train_indices, val_indices, stats

def validate(model, val_loader, device, num_batches):
    model.eval()
    val_loss = 0
    val_iou_loss = 0
    val_angle_loss = 0
    
    with torch.no_grad():
        with tqdm(total=num_batches, desc='Validating') as pbar:
            for img, gt_score_map, gt_geo_map, roi_mask in val_loader:
                img, gt_score_map, gt_geo_map, roi_mask = (img.to(device),
                    gt_score_map.to(device), gt_geo_map.to(device), roi_mask.to(device))
                
                loss, extra_info = model.train_step(img, gt_score_map, gt_geo_map, roi_mask)
                
                val_loss += loss.item()
                val_iou_loss += extra_info['iou_loss']
                val_angle_loss += extra_info['angle_loss']
                
                pbar.update(1)
                pbar.set_postfix({
                    'Val Loss': f'{loss.item():.4f}',
                    'IoU Loss': f'{extra_info["iou_loss"]:.4f}',
                    'Angle Loss': f'{extra_info["angle_loss"]:.4f}'
                })
    
    num_samples = len(val_loader)
    return (val_loss / num_samples, 
            val_iou_loss / num_samples,
            val_angle_loss / num_samples)


def do_training(data_dir, model_dir, device, image_size, input_size, num_workers, batch_size,
                learning_rate, max_epoch, save_interval, wandb_project, wandb_name, wandb_entity,
                fold=0, n_folds=5):
    # WandB 초기화
    wandb.init(
        project=wandb_project, 
        name=f"{wandb_project}_fold{fold}")
    
    # 기본 데이터셋 생성
    base_dataset = SceneTextDataset(
        data_dir,
        split='train',
        image_size=image_size,
        crop_size=input_size,
    )
    
    # K-fold 분할
    train_indices, val_indices, stats = create_fold_datasets(base_dataset, fold, n_folds)
    
    train_dataset = Subset(base_dataset, train_indices)
    val_dataset = Subset(base_dataset, val_indices)

    # EAST 데이터셋으로 변환
    train_dataset = EASTDataset(train_dataset)
    val_dataset = EASTDataset(val_dataset)

    # DataLoader 설정
    train_num_batches = math.ceil(len(train_dataset) / batch_size)
    val_num_batches = math.ceil(len(val_dataset) / batch_size)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    # 기존 코드를 아래와 같이 변경
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = EAST()  
    model.to(device)
    
    # AdamW 옵티마이저
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01
    )
    
    # CosineAnnealingLR 스케줄러
    scheduler = lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max_epoch,  # 전체 에폭 수
        eta_min=1e-6  # 최소 learning rate
    )

    best_metric = float('inf')
    patience = 20
    patience_counter = 0
    
    model.train()
    for epoch in range(max_epoch):
        epoch_loss = 0
        epoch_iou_loss = 0  
        epoch_angle_loss = 0  
        epoch_start = time.time()
        
        with tqdm(total=train_num_batches) as pbar:
            for img, gt_score_map, gt_geo_map, roi_mask in train_loader:
                pbar.set_description('[Epoch {}]'.format(epoch + 1))

                try:
                    loss, extra_info = model.train_step(img, gt_score_map, gt_geo_map, roi_mask)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    # loss 값 누적
                    epoch_loss += loss.item()
                    if extra_info['iou_loss'] is not None:
                        epoch_iou_loss += extra_info['iou_loss']
                    if extra_info['angle_loss'] is not None:
                        epoch_angle_loss += extra_info['angle_loss']

                    pbar.update(1)
                    pbar.set_postfix({
                        'Loss': f'{loss.item():.4f}',
                        'IoU': f'{extra_info["iou_loss"]:.4f}' if extra_info['iou_loss'] is not None else 'N/A',
                        'Angle': f'{extra_info["angle_loss"]:.4f}' if extra_info['angle_loss'] is not None else 'N/A'
                    })
                except Exception as e:
                    print(f"\nError in training step: {str(e)}")
                    print(f"Shape of inputs - Image: {img.shape}, Score Map: {gt_score_map.shape}, "
                          f"Geo Map: {gt_geo_map.shape}, ROI Mask: {roi_mask.shape}")
                    continue

        scheduler.step()

        # Validation
        val_loss, val_iou_loss, val_angle_loss = validate(model, val_loader, device, val_num_batches)
        
        # Metrics 계산 (배치 수로 나누어 평균 계산)
        train_loss = epoch_loss / train_num_batches
        train_iou_loss = epoch_iou_loss / train_num_batches
        train_angle_loss = epoch_angle_loss / train_num_batches
        
        # 현재 성능 metric 계산 (IoU + Angle loss)
        current_metric = val_iou_loss + 0.5 * val_angle_loss
        
        # Logging
        wandb.log({
            "train_loss": train_loss,
            "train_iou_loss": train_iou_loss,
            "train_angle_loss": train_angle_loss,
            "val_loss": val_loss,
            "val_iou_loss": val_iou_loss,
            "val_angle_loss": val_angle_loss,
            "learning_rate": optimizer.param_groups[0]['lr']
        })
        
        print(f'Epoch {epoch+1}/{max_epoch}:')
        print(f'Train - Loss: {train_loss:.4f}, IoU: {train_iou_loss:.4f}, Angle: {train_angle_loss:.4f}')
        print(f'Val - Loss: {val_loss:.4f}, IoU: {val_iou_loss:.4f}, Angle: {val_angle_loss:.4f}')
        print(f'Time: {timedelta(seconds=time.time()-epoch_start)}')
        
        # 모델 저장 디렉토리 생성
        if not osp.exists(model_dir):
            os.makedirs(model_dir)
            
        # Best model 저장
        if current_metric < best_metric:
            best_metric = current_metric
            patience_counter = 0
            save_path = osp.join(model_dir, f'best_model_fold{fold}.pth')
            torch.save(model.state_dict(), save_path)
            print(f'New best model saved! (Val IoU: {val_iou_loss:.4f}, Val Angle: {val_angle_loss:.4f})')
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= patience:
            print(f'Early stopping triggered after epoch {epoch+1}')
            break
        
        # 5 에폭마다 모델 저장 (latest_epoch{N})
        if (epoch + 1) % save_interval == 0:  # save_interval(5) 에폭마다 저장
            ckpt_fpath = osp.join(model_dir, f'latest_epoch{epoch+1}_fold{fold}.pth')
            torch.save(model.state_dict(), ckpt_fpath)
            print(f'Model saved: latest_epoch{epoch+1}_fold{fold}.pth')

    wandb.finish()  


def main(args):
    do_training(**args.__dict__)

if __name__ == '__main__':
    args = parse_args()
    main(args)