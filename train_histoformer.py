import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import os
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as PSNR

from basicsr.models.archs.histoformer_arch import Histoformer
from basicsr.models.losses import L1Loss
from loss.loss import  PerceptualLoss
from data.dataset import Pansharpening_mat_Dataset, MRI_pre_dataset, NYU_v2_datset, MultiTaskDataset
from torchvision import transforms
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter

from tools.logger import *
from tools.evaluation_metric import calc_rmse
from basicsr.models import lr_scheduler as lr_scheduler


def get_opt():
    parser = argparse.ArgumentParser(description='Hyper-parameters for network')
    parser.add_argument('--exp_name', type=str, default='Histoformer', help='experiment name')
    parser.add_argument('-learning_rate', help='Set the learning rate', default=3e-4, type=float)
    parser.add_argument('-batch_size', help='Set the training batch size', default=2, type=int)
    parser.add_argument('-epoch_start', help='Starting epoch number of the training', default=0, type=int)
    parser.add_argument('-seed', help='set random seed', default=19, type=int)
    parser.add_argument('-num_epochs', help='', default=600, type=int)
    parser.add_argument('-depth_root', help='', default='/data/cjj/dataset/NYU_V2', type=str)
    parser.add_argument('-mri_root', help='', default='/data/wtt/MRI_align/BT', type=str)
    parser.add_argument('-pan_root', help='', default='/data/datasets/pansharpening/NBU_dataset0730', type=str)
    parser.add_argument('-save_dir', help='', default='/home/cjj/projects/AIO_compare/Histoformer/Checkpoint', type=str)
    parser.add_argument('-gpu_id', help='', default=0, type=int)
    
    args = parser.parse_args()
    
    return args

def train_one_epoch(epoch, dataloader, model, optimizer, cri_pix, cri_perceptual, writer):
    try:
        model.train()
        epoch_losses = {'total': 0, 'task_losses': [[] for _ in range(9)]}
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        for train_data_lists in pbar:
            for task_id, data_list in enumerate(train_data_lists):
                inp_lr, inp_gt, inp_guide = (item.type(torch.FloatTensor).cuda() for item in data_list)

                # --- Zero the parameter gradients --- #
                optimizer.zero_grad()

                # --- Forward + Backward + Optimize --- #
                output = model(inp_lr, inp_guide)
                
                pix_loss = cri_pix(output, inp_gt)
                if inp_gt.shape[1] == 4:
                    rgb_output = output[:,0:3,:,:]
                    rgb_gt = inp_gt[:,0:3,:,:]
                    ir_output = output[:,3:4,:,:].repeat(1,3,1,1)
                    ir_gt = inp_gt[:,3:4,:,:].repeat(1,3,1,1)
                    l_percep_rgb, _ = cri_perceptual(rgb_output, rgb_gt)
                    l_percep_ir, _ =cri_perceptual(ir_output, ir_gt)
                    l_percep = l_percep_rgb + l_percep_ir
                else:
                    output = output.repeat(1,3,1,1)
                    gt = inp_gt.repeat(1,3,1,1)
                    l_percep, _ = cri_perceptual(output, gt)
                
                loss = pix_loss + l_percep 
                loss.backward()
                optimizer.step()

                epoch_losses['total'] += loss.item()
                epoch_losses['task_losses'][task_id].append(loss.item())

                writer.add_scalar(f'Train/Task_{task_id}_Loss', loss.item(), epoch)

                pbar.set_postfix(
                        loss=loss.item(),
                        task=task_id,
                        lr=optimizer.param_groups[0]['lr']
                    )
            
        avg_total_loss = epoch_losses['total'] / len(dataloader)
        avg_task_losses = [np.mean(losses) if losses else 0 for losses in epoch_losses['task_losses']]
        
        return avg_total_loss, avg_task_losses
    finally:
        # 确保数据加载器被正确清理
        if hasattr(dataloader, '_iterator'):
            del dataloader._iterator

def validate_one_epoch(model, datasets, test_minmax, logger, epoch, writer, save_dir, optimizer, scheduler):
    global best_psnr_pan_WV4, best_psnr_pan_QB, best_psnr_pan_GF1
    global best_psnr_mri_2x, best_psnr_mri_4x, best_psnr_mri_8x
    global best_rmse_4, best_rmse_8, best_rmse_16
    
    model.eval()
    with torch.no_grad():
        psnr = []
        rmse = []

        best_metrics = {
            'rmse_4': best_rmse_4,
            'rmse_8': best_rmse_8,
            'rmse_16': best_rmse_16,
            'psnr_mri_2x': best_psnr_mri_2x,
            'psnr_mri_4x': best_psnr_mri_4x,
            'psnr_mri_8x': best_psnr_mri_8x,
            'psnr_pan_wv4': best_psnr_pan_WV4,
            'psnr_pan_qb': best_psnr_pan_QB,
            'psnr_pan_gf1': best_psnr_pan_GF1
        }

        for dataset_id, dataset in enumerate(datasets):
            val_dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
            metric_values = []
            if dataset_id <= 2:  # depth estimation tasks
                for index, (inp_lr, inp_gt, inp_guide) in enumerate(tqdm(val_dataloader, desc=f'Validating Depth_{4*(2**dataset_id)}')):
                    output = model(inp_lr.cuda(), inp_guide.cuda())[0]
                    metric_values.append(
                        calc_rmse(output[0], 
                                inp_gt.cuda()[0,0], 
                                torch.from_numpy(test_minmax[:, index]).cuda())
                    )
                avg_metric = torch.mean(torch.stack(metric_values)).item()
                rmse.append(avg_metric)
            
            else:  # MRI and pansharpening tasks
                for inp_lr, inp_gt, inp_guide in tqdm(val_dataloader, desc=f'Validating Task_{dataset_id}'):
                    output = model(inp_lr.cuda(), inp_guide.cuda())
                    metric_values.append(
                        PSNR(inp_gt.numpy()[0], output.cpu().numpy()[0])
                    )
                avg_metric = np.mean(metric_values)
                psnr.append(avg_metric)
        
        # 记录验证指标
        metrics = {
            'rmse_4': rmse[0],
            'rmse_8': rmse[1],
            'rmse_16': rmse[2],
            'psnr_mri_2x': psnr[0],
            'psnr_mri_4x': psnr[1],
            'psnr_mri_8x': psnr[2],
            'psnr_pan_wv4': psnr[3],
            'psnr_pan_qb': psnr[4],
            'psnr_pan_gf1': psnr[5]
        }
        
        # 记录到tensorboard
        for name, value in metrics.items():
            writer.add_scalar(f'Validation/{name.upper()}', value, epoch)
        
        # 检查并保存最佳模型
        improved = False
        for metric_name, metric_value in metrics.items():
            is_better = metric_value > best_metrics[metric_name] if 'psnr' in metric_name else metric_value < best_metrics[metric_name]
            if is_better:
                improved = True
                best_metrics[metric_name] = metric_value
                globals()[f'best_{metric_name}'] = metric_value
                model_path = os.path.join(save_dir, f'Best_{metric_name.upper()}.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'metric_value': metric_value,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                }, model_path)
        
        # 记录所有指标
        log_message = f'Epoch {epoch} Validation:\n'
        for name, value in metrics.items():
            log_message += f'{name}: {value:.4f}, '
        logger.info(log_message)
        
        return metrics, improved
    
def main(opt):

    os.makedirs(opt.save_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(opt.save_dir, 'tensorboard_logs'))

    logger = get_logger(os.path.join(opt.save_dir, 'run.log'))
    logger.info(opt)

    # 在这里定义全局变量
    global best_psnr_pan_WV4, best_psnr_pan_QB, best_psnr_pan_GF1
    global best_psnr_mri_2x, best_psnr_mri_4x, best_psnr_mri_8x
    global best_rmse_4, best_rmse_8, best_rmse_16

    # Initialize best metrics
    best_psnr_pan_WV4, best_psnr_pan_QB, best_psnr_pan_GF1 = -float('inf'), -float('inf'), -float('inf')
    best_psnr_mri_2x, best_psnr_mri_4x, best_psnr_mri_8x = -float('inf'), -float('inf'), -float('inf')
    best_rmse_4, best_rmse_8, best_rmse_16 = float('inf'), float('inf'), float('inf')

    # Datasets and DataLoader
    data_transform = transforms.Compose([transforms.ToTensor()])

    # Training dataset
    depth_dataset_4 = NYU_v2_datset(root_dir=opt.depth_root, scale=4, transform=data_transform, train=True)
    depth_dataset_8 = NYU_v2_datset(root_dir=opt.depth_root, scale=8, transform=data_transform, train=True)
    depth_dataset_16 = NYU_v2_datset(root_dir=opt.depth_root, scale=16, transform=data_transform, train=True)
    mri_dataset_2 = MRI_pre_dataset(os.path.join(opt.mri_root, 'x2_t2_train'), os.path.join(opt.mri_root, 'T2_train'), os.path.join(opt.mri_root, 'T1_train'))
    mri_dataset_4 = MRI_pre_dataset(os.path.join(opt.mri_root, 'x4_t2_train'), os.path.join(opt.mri_root, 'T2_train'), os.path.join(opt.mri_root, 'T1_train'))
    mri_dataset_8 = MRI_pre_dataset(os.path.join(opt.mri_root, 'x8_t2_train'), os.path.join(opt.mri_root, 'T2_train'), os.path.join(opt.mri_root, 'T1_train'))
    pan_dataset_WV4 = Pansharpening_mat_Dataset(os.path.join(opt.pan_root, 'WV4', 'train'))
    pan_dataset_QB = Pansharpening_mat_Dataset(os.path.join(opt.pan_root, 'QB', 'train'))
    pan_dataset_GF1 = Pansharpening_mat_Dataset(os.path.join(opt.pan_root, 'GF1', 'train'))
    
    mix_dataset = MultiTaskDataset(depth_dataset_4, depth_dataset_8, depth_dataset_16, 
                                  mri_dataset_2, mri_dataset_4, mri_dataset_8,
                                  pan_dataset_WV4, pan_dataset_QB, pan_dataset_GF1)
    
    # Validation Datasets
    test_minmax = np.load(f'{opt.depth_root}/test_minmax.npy')
    val_depth_dataset_4 = NYU_v2_datset(root_dir=opt.depth_root, scale=4, transform=data_transform, train=False)
    val_depth_dataset_8 = NYU_v2_datset(root_dir=opt.depth_root, scale=8, transform=data_transform, train=False)
    val_depth_dataset_16 = NYU_v2_datset(root_dir=opt.depth_root, scale=16, transform=data_transform, train=False)
    val_mri_dataset_2 = MRI_pre_dataset(os.path.join(opt.mri_root, 'x2_t2_test'), os.path.join(opt.mri_root, 'T2_test'), os.path.join(opt.mri_root, 'T1_test'))
    val_mri_dataset_4 = MRI_pre_dataset(os.path.join(opt.mri_root, 'x4_t2_test'), os.path.join(opt.mri_root, 'T2_test'), os.path.join(opt.mri_root, 'T1_test'))
    val_mri_dataset_8 = MRI_pre_dataset(os.path.join(opt.mri_root, 'x8_t2_test'), os.path.join(opt.mri_root, 'T2_test'), os.path.join(opt.mri_root, 'T1_test'))
    val_pan_dataset_WV4 = Pansharpening_mat_Dataset(os.path.join(opt.pan_root,'WV4', 'test'))
    val_pan_dataset_QB = Pansharpening_mat_Dataset(os.path.join(opt.pan_root,'QB', 'test'))
    val_pan_dataset_GF1 = Pansharpening_mat_Dataset(os.path.join(opt.pan_root,'GF1', 'test'))

    list_val_dataset = [val_depth_dataset_4, val_depth_dataset_8, val_depth_dataset_16, 
                        val_mri_dataset_2, val_mri_dataset_4, val_mri_dataset_8,
                        val_pan_dataset_WV4, val_pan_dataset_QB, val_pan_dataset_GF1
                        ]
    

    # --- Define the network --- #
    Generator = Histoformer().cuda()
    optimizer_G = torch.optim.AdamW(Generator.parameters(), lr=opt.learning_rate,betas=[0.9,0.999], weight_decay=1e-4)
    lr_scheduler_G = lr_scheduler.CosineAnnealingRestartCyclicLR(optimizer_G, periods=[92000, 208000], restart_weights=[1,1], eta_mins=[0.0003,0.000001])

    # Loss
    cri_perceptual = PerceptualLoss(layer_weights={'conv5_4': 1}, vgg_type="vgg19", use_input_norm=True,
    range_norm=False, perceptual_weight=0.1, style_weight=0,criterion="l1")
    cri_pix = L1Loss(loss_weight = 1.0, reduction="mean")

    # logger.info("Running initial validation...")
    # metrics, improved = validate_one_epoch(
    #     Generator, list_val_dataset, test_minmax, logger, 0,
    #     writer, opt.save_dir, optimizer_G, lr_scheduler_G
    # )
    # logger.info("Initial validation completed.")

    Generator.train()

    for epoch in range(opt.epoch_start, opt.num_epochs):
        mix_dataset.shuffle()
        train_dataloader = DataLoader(
            mix_dataset, 
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            persistent_workers=False,
            prefetch_factor=2
        )

        try:
            avg_total_loss, avg_task_losses = train_one_epoch(
                epoch, train_dataloader, Generator, optimizer_G, cri_pix, cri_perceptual, writer)
            
            lr_scheduler_G.step()

            if epoch <= 250 and epoch % 15 == 0:
                metrics, improved = validate_one_epoch(Generator, list_val_dataset, test_minmax, 
                                    logger, epoch, writer, opt.save_dir, optimizer_G, lr_scheduler_G)
            elif epoch > 250 and epoch % 7 == 0:
                metrics, improved = validate_one_epoch(Generator, list_val_dataset, test_minmax, 
                                    logger, epoch, writer, opt.save_dir, optimizer_G, lr_scheduler_G)
                
                checkpoint = {
                'epoch': epoch,
                'model_state_dict': Generator.state_dict(),
                'optimizer_state_dict': optimizer_G.state_dict(),
                'scheduler_state_dict': lr_scheduler_G.state_dict(),
                'metrics': metrics
            }
                torch.save(checkpoint, os.path.join(opt.save_dir, 'latest_checkpoint.pth'))
                if improved:
                    torch.save(checkpoint, os.path.join(opt.save_dir, f'checkpoint_epoch_{epoch}.pth'))

        finally:
            # 确保数据加载器被正确清理
            del train_dataloader
            torch.cuda.empty_cache()
    writer.close()

if __name__ == '__main__':
    opt = get_opt()
    torch.cuda.set_device(opt.gpu_id)
    main(opt)