import os
import argparse
import numpy as np
from tqdm import tqdm
import logging
from glob import glob
from pandas import DataFrame, Series
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import heapq
from scipy.stats import entropy
import pandas as pd

from utils import setup_seed, cos_sim
from model.adapter import AdaptedCLIP
from model.clip import create_model
from dataset import get_dataset, DOMAINS
from forward_utils import (
    get_adapted_text_embedding,
    calculate_similarity_map,
    metrics_eval,
    visualize,
)
import warnings

warnings.filterwarnings("ignore")

cpu_num = 4

os.environ["OMP_NUM_THREADS"] = str(cpu_num)
os.environ["OPENBLAS_NUM_THREADS"] = str(cpu_num)
os.environ["MKL_NUM_THREADS"] = str(cpu_num)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(cpu_num)
os.environ["NUMEXPR_NUM_THREADS"] = str(cpu_num)
torch.set_num_threads(cpu_num)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def get_support_features(model, support_loader, device):
    all_features = []
    for input_data in support_loader:  # bs always=1. training for an epoch first, Then use this updated model for memory bank construction.
        image = input_data[0].to(device)
        patch_tokens = model(image)
        patch_tokens = [t.reshape(-1, 768) for t in patch_tokens]
        all_features.append(patch_tokens)
    support_features = [
        torch.cat([all_features[j][i] for j in range(len(all_features))], dim=0)
        for i in range(len(all_features[0]))
    ]
    return support_features
def get_predictions(
    model: nn.Module,
    class_text_embeddings: torch.Tensor,
    test_loader: DataLoader,
    device: str,
    img_size: int,
    dataset: str = "MVTec",
):
    masks = []
    labels = []
    preds = []
    preds_image = []
    file_names = []
    for input_data in tqdm(test_loader):
        image = input_data["image"].to(device)
        # print("image.shape",image.shape) #批量x3x518x518
        mask = input_data["mask"].cpu().numpy()
        # print("mask.shape",mask.shape) #批量x3x518x518
        label = input_data["label"].cpu().numpy()
        # print("label.shape",label.shape) #批量
        file_name = input_data["file_name"] 
        # print("file_name",file_name) #
        '''' file_name ['images/208.png', 'images/192.png', 'images/160.png', 'images/171.png', 'images/180.png', 'images/150.png', 
        'images/154.png', 'images/174.png', 'images/189.png', 'images/153.png', 'images/157.png', 'images/200.png', 'images/182.png',
        'images/198.png', 'images/177.png', 'images/175.png', 'images/181.png', 'images/169.png', 'images/163.png', 'images/201.png', '
        images/178.png', 'images/207.png', 'images/158.png', 'images/165.png', 'images/199.png', 'images/172.png', 'images/164.png', 
        'images/176.png', 'images/167.png', 'images/193.png', 'images/185.png', 'images/195.png']'''
        # print("len(file_name)",len(file_name)) #批量
        # set up class-specific containers
        class_name = input_data["class_name"]
        # print("class_name",class_name) #CVC-300
        assert len(set(class_name)) == 1, "mixed class not supported"
        masks.append(mask)
        labels.append(label)
        file_names.extend(file_name)
        # print("len(file_names)",len(file_names)) #批量
        # get text 
        epoch_text_feature = class_text_embeddings
        # print("epoch_text_feature",epoch_text_feature.shape) #768x2
        # forward image 
        # 获取分割和检测特征
        patch_features, det_feature= model(image)
        # print(" patch_features.shape",patch_features[0].shape) #批量x1369x768
        # print("det_feature.shape",det_feature.shape) #批量x1369x768
        det_feature = F.normalize(det_feature, dim=-1).mean(1)
        # print("det_feature.shape",det_feature.shape) #批量x768
            
        # print("len(seg_features)",len(patch_features)) # 4
        # print("seg_feature",patch_features[0].shape) #批量x1369x768
        # print("det_feature",det_feature.shape) #批量x768
        # calculate similarity and get prediction
        # cls_preds = []
        #与文本特征计算相似度得到异常分类预测
        pred = det_feature @ epoch_text_feature
        # print("pred.shape",pred.shape) #批量x2
        pred = (pred[:, 1] + 1) / 2
        # print("pred.shape_cated",pred.shape) #批量
        preds_image.append(pred.cpu().numpy())
        # print("len(preds_image)",len(preds_image)) #2
        
        patch_preds = []
        for f in patch_features:
            # f: bs,patch_num,768
            # print("f.shape",f.shape) #批量x1369x768
            patch_pred = calculate_similarity_map(
                f, epoch_text_feature, img_size, test=True, domain=DOMAINS[dataset]
            )
            # print("patch_pred.shape",patch_pred.shape) #批量x1x518x518
            patch_preds.append(patch_pred)
            # print("len(patch_preds)",len(patch_preds)) #4
            
        patch_preds = torch.cat(patch_preds, dim=1).sum(1).cpu().numpy()
        # print("patch_preds.shape",patch_pred.shape) #批量x1x518x518
        
        preds.append(patch_preds)
        # print("len(patch_preds)",len(patch_preds)) #批量
        
    masks = np.concatenate(masks, axis=0)
    # print("masks.shape",masks.shape) #类别总图片数量x1x518x518
    labels = np.concatenate(labels, axis=0)
    # print("labels.shape",labels.shape) #类别总图片数量
    preds = np.concatenate(preds, axis=0)
    # print("preds.shape",preds.shape) #类别总图片数量x518x518
    preds_image = np.concatenate(preds_image, axis=0)
    # print("preds_image.shape",preds_image.shape) # 类别总图片数量
    return masks, labels, preds, preds_image, file_names


def main():
    parser = argparse.ArgumentParser(description="Training")
    # model
    parser.add_argument(
        "--model_name",
        type=str,
        default="ViT-L-14-336",
        help="ViT-B-16-plus-240, ViT-L-14-336",
    )
    parser.add_argument("--img_size", type=int, default=518)
    parser.add_argument("--relu", action="store_true")
    # testing
    parser.add_argument("--dataset", type=str, default="MVTec")
    parser.add_argument("--shot", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=32)
    # exp
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--save_path", type=str, default="ckpt/baseline")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--text_norm_weight", type=float, default=0.1)
    parser.add_argument("--text_adapt_weight", type=float, default=0.1)
    parser.add_argument("--image_adapt_weight", type=float, default=0.1)
    parser.add_argument("--text_adapt_until", type=int, default=3)
    parser.add_argument("--image_adapt_until", type=int, default=6)

    args = parser.parse_args()
    # ========================================================
    setup_seed(args.seed)
    # check save_path and setting logger
    os.makedirs(args.save_path, exist_ok=True)
    logger = logging.getLogger(__name__)

    logger = logging.getLogger(__name__)
    logging.basicConfig(
        filename=os.path.join(args.save_path, "test.log"),
        encoding="utf-8",
        level=logging.INFO,
    )
    logger.info("args: %s", vars(args))
    # set device
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    # ========================================================
    # load model
    # set up model for testing
    clip_model = create_model(
        model_name=args.model_name,
        img_size=args.img_size,
        device=device,
        pretrained="openai",
        require_pretrained=True,
    )
    clip_model.eval()
    #定义Adapter模型
    model = AdaptedCLIP(
        clip_model=clip_model,
        text_adapt_weight=args.text_adapt_weight,
        image_adapt_weight=args.image_adapt_weight,
        text_adapt_until=args.text_adapt_until,
        image_adapt_until=args.image_adapt_until,
        relu=args.relu,
    ).to(device)
    model.eval()
    # load checkpoints if exists
    text_file = glob(args.save_path + "/text_adapter.pth")
    assert len(text_file) >= 0, "text adapter checkpoint not found"
    if len(text_file) > 0:
        checkpoint = torch.load(text_file[0])
        model.text_adapter.load_state_dict(checkpoint["text_adapter"])
        adapt_text = True
    else:
        adapt_text = False

    files = sorted(glob(args.save_path + "/image_adapter_15.pth"))
    assert len(files) > 0, "image adapter checkpoint not found"
    for file in files:
        checkpoint = torch.load(file)
        model.image_adapter.load_state_dict(checkpoint["image_adapter"])
        test_epoch = checkpoint["epoch"]
        logger.info("-----------------------------------------------")
        logger.info("load model from epoch %d", test_epoch)
        logger.info("-----------------------------------------------")
        # ========================================================
        # load dataset
        kwargs = {"num_workers": 4, "pin_memory": True} if use_cuda else {}
        image_datasets = get_dataset(
            args.dataset,
            args.img_size,
            None,
            args.shot,
            "test",
            logger=logger,
        )
        #设置文本嵌入adapter
        with torch.no_grad():
            if adapt_text:
                text_embeddings = get_adapted_text_embedding(
                    model, args.dataset, device
                )
            else:
                text_embeddings = get_adapted_text_embedding(
                    clip_model, args.dataset, device
                )
        # ========================================================
        df = DataFrame(
            columns=[
                "class name",
                "pixel AUC",
                "pixel AP",
                "image AUC",
                "image AP",
            ]
        )
        #对每个类别分别计算指标
        for class_name, image_dataset in image_datasets.items():
            image_dataloader = torch.utils.data.DataLoader(
                image_dataset, batch_size=args.batch_size, shuffle=False, **kwargs
            )

            # ========================================================
            # testing
            with torch.no_grad():
                class_text_embeddings = text_embeddings[class_name]
                masks, labels, preds, preds_image, file_names = get_predictions(
                    model=model,
                    class_text_embeddings=class_text_embeddings,
                    test_loader=image_dataloader,
                    device=device,
                    img_size=args.img_size,
                    dataset=args.dataset,
                )
                
                
            # ========================================================
            if args.visualize:
                visualize(
                    masks,
                    preds,
                    file_names,
                    args.save_path,
                    args.dataset,
                    class_name=class_name,
                )
            
            class_result_dict = metrics_eval(
                masks,
                labels,
                preds,
                preds_image,
                class_name,
                domain=DOMAINS[args.dataset],
            ) 
            #每个类别加入一个Series到DataFrame中
            df.loc[len(df)] = Series(class_result_dict)
         
        # 确保数值列都是数字类型
        numeric_cols = ['pixel AUC', 'pixel AP', 'image AUC', 'image AP']
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

        # 处理特殊值（如0值表示缺失的情况）
        df[numeric_cols] = df[numeric_cols].replace(0, np.nan)

        # 计算平均值（自动忽略NaN）
        avg_values = df[numeric_cols].mean()

        # 添加新行
        df.loc[len(df)] = avg_values
        df.loc[len(df)-1, 'class name'] = 'Average'
        logger.info("final results:\n%s", df.to_string(index=False, justify="center"))


if __name__ == "__main__":
    main()
