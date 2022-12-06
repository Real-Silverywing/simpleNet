import pandas as pd
import torch
import albumentations as A
from utils.kpdataloader import KeyPointDataset
from albumentations.pytorch import ToTensorV2
from utils.cv2dataloader import VideoFolder
# from utils.torch_videovision.videotransforms import video_transforms, volume_transforms
from utils.cv2dataloader import get_data_loaders
# from utils.segment_dataloader import get_data_loaders as get_segment_dataloaders
from utils.kpdataloader import get_data_loaders as get_kpdataloaders
from utils.cat_kpdataloader import get_data_loaders as get_cat_kpdataloaders
# from feature_dataloader import get_data_loaders as get_featdataloaders
# from mixed_dataloader import get_data_loaders as get_mixeddataloaders
# from preprocessed_dataloader import get_data_loaders as get_processed_data_loaders

def get_ky_dataloaders_from_config(config, TEST_CSV_PATH, num_folds=4):
    temp = pd.read_csv(TEST_CSV_PATH, header=0)
    test_labels = temp['Label'].tolist()
    temp = temp['Dir'].str.split(" ", n = 1, expand = True)
    test_ids = temp[0].tolist()
    test_df = pd.DataFrame({0: test_ids, 1: test_labels})

    if config.dataloader == "cv2dataloader":    
        video_transform_list = [
                                video_transforms.RandomRotation(30),
                                video_transforms.RandomHorizontalFlip(),
                                video_transforms.Resize(config.imsize),
                                video_transforms.RandomCrop(config.imsize),
                                volume_transforms.ClipToTensor()
                                ]
        transform = video_transforms.Compose(video_transform_list)
        test_loader = VideoFolder(root="./data/",
                            input=test_df,
                            clip_size=config.clip_size,
                            nclips=config.nclips,
                            step_size=config.step_size,
                            is_val=True,
                            transform=transform,
                            )
        test_loader = torch.utils.data.DataLoader(test_loader,
                                batch_size=config.batch_size, shuffle=False,
                                num_workers=config.num_workers, pin_memory=True)
    elif config.dataloader=="kpdataloader":
        print("DEBUG: getting kpdataloader")
        cropsize = int(config.imsize*0.875)
        val_transform = A.ReplayCompose([
            A.Resize(height=config.imsize, width=config.imsize),
            A.CenterCrop(width=cropsize, height=cropsize),
            A.transforms.Normalize(),
            ToTensorV2(),
        ],keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
        test_loader = KeyPointDataset(
                        input=test_df,
                        clip_size=config.clip_size,
                        nclips=config.nclips,
                        step_size=config.step_size,
                        is_val=True,
                        transform=val_transform,
                        )
        test_loader = torch.utils.data.DataLoader(test_loader,
                                batch_size=config.batch_size, shuffle=False,
                                num_workers=config.num_workers, pin_memory=True)
    elif config.dataloader=="mixed_dataloader":
        print("DEBUG: getting mixed dataloader")
        _, _, test_loader = get_mixeddataloaders(
            config=config,
            val_fold=1,
            imsize=config.imsize,
            batch_size=config.batch_size,
            clip_size=config.clip_size,
            nclips=config.nclips,
            step_size=config.step_size,
            num_workers=config.num_workers,
            dum=False,
            num_folds=num_folds,
        )

    return test_loader



def get_dataloaders_from_config(config, val_fold, test_fold, dum=False, num_folds=4):
    if config.dataloader == "preprocessed_dataloader":
        train_loader, valid_loader, test_loader = get_processed_data_loaders(
            config=config,
            val_fold=val_fold,
            test_fold=test_fold,
            batch_size=config.batch_size,
            num_segments=config.num_segments,
            size_snippets=config.size_snippets,
            num_workers=config.num_workers,
            dum=dum,
            num_folds=num_folds,
        )
    elif config.dataloader == "cv2dataloader":
        train_loader, valid_loader, test_loader = get_data_loaders(
            config=config,
            val_fold=val_fold,
            test_fold=test_fold,
            imsize=config.imsize,
            batch_size=config.batch_size,
            clip_size=config.clip_size,
            nclips=config.nclips,
            step_size=config.step_size,
            num_workers=config.num_workers,
            dum=dum,
            num_folds=num_folds,
        )
    elif config.dataloader == "segment_dataloader":
        print("DEBUG: getting segment dataloader")
        train_loader, valid_loader, test_loader = get_segment_dataloaders(
            config=config,
            val_fold=val_fold,
            test_fold=test_fold,
            batch_size=config.batch_size,
            imsize=config.imsize,
            num_segments=config.num_segments,
            size_snippets=config.size_snippets,
            num_workers=config.num_workers,
            dum=dum,
            num_folds=num_folds,
        )
    elif config.dataloader == "kpdataloader":
        print("DEBUG: getting kpdataloader")
        train_loader, valid_loader, test_loader = get_kpdataloaders(
            config=config,
            val_fold=val_fold,
            test_fold=test_fold,
            imsize=config.imsize,
            batch_size=config.batch_size,
            clip_size=config.clip_size,
            nclips=config.nclips,
            step_size=config.step_size,
            num_workers=config.num_workers,
            dum=dum,
            num_folds=num_folds,
        )
    elif config.dataloader == "cat_kpdataloader":
        print("DEBUG: getting kpdataloader")
        train_loader, valid_loader, test_loader = get_cat_kpdataloaders(
            config=config,
            val_fold=val_fold,
            test_fold=test_fold,
            imsize=config.imsize,
            batch_size=config.batch_size,
            clip_size=config.clip_size,
            nclips=config.nclips,
            step_size=config.step_size,
            num_workers=config.num_workers,
            dum=dum,
            num_folds=num_folds,
        )
    elif config.dataloader=="feature_dataloader":
        print("DEBUG: getting feature_dataloader")
        train_loader, valid_loader, test_loader = get_featdataloaders(
            config=config,
            val_fold=val_fold,
            test_fold=test_fold,
            step_size=config.step_size,
            num_workers=config.num_workers,
            dum=False,
            num_folds=num_folds,
        )
    elif config.dataloader=="mixed_dataloader":
        print("DEBUG: getting mixed_dataloader")
        train_loader, valid_loader, test_loader = get_mixeddataloaders(
            config=config,
            val_fold=val_fold,
            imsize=config.imsize,
            batch_size=config.batch_size,
            clip_size=config.clip_size,
            nclips=config.nclips,
            step_size=config.step_size,
            num_workers=config.num_workers,
            dum=dum,
            num_folds=num_folds,
        )
    
    return train_loader, valid_loader, test_loader
