from datetime import datetime
from pathlib import Path

import torch
import torch.utils.tensorboard as tb

from .models import load_model, save_model
from .metrics import DetectionMetric
from .datasets.road_dataset import load_data
import time


class ClassificationLoss(torch.nn.Module):
    def forward(self, logits: torch.Tensor, target: torch.LongTensor) -> torch.Tensor:
        """
        Multi-class classification loss

        Args:
            logits: tensor (b, c) logits, where c is the number of classes
            target: tensor (b,) labels

        Returns:
            tensor, scalar loss
        """
        return torch.nn.functional.cross_entropy(logits, target)
    
class RegressionLoss(torch.nn.Module):
    def forward(self, logits: torch.Tensor, target: torch.LongTensor) -> torch.Tensor:
        """
        Multi-class classification loss

        Args:
            logits: tensor (b,) logits
            target: tensor (b,) labels

        Returns:
            tensor, scalar loss
        """
        return torch.nn.functional.l1_loss(logits, target) + torch.nn.functional.mse_loss(logits, target)
    
def train(
    exp_dir: str = "logs",
    model_name: str = "detector",
    num_epoch: int = 5,
    lr: float = 1e-2,
    batch_size: int = 32,
    seed: int = 2024,
):
    print(f"{model_name=} {num_epoch=} {lr=} {batch_size=}")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        print("CUDA not available, using CPU")
        device = torch.device("cpu")

    # set random seed so each run is deterministic
    torch.manual_seed(seed)

    # directory with timestamp to save tensorboard logs and model checkpoints
    log_dir = Path(exp_dir) / f"{model_name}_{datetime.now().strftime('%m%d_%H%M%S')}"
    logger = tb.SummaryWriter(log_dir)

    model = load_model(model_name)
    model = model.to(device)
    model.train() # inidcates to tensor that we are training. some behavior is different compared to eval()
    # num_workers=0 -> time = 351 s for one epoch
    # num_workers=2 -> time = 343 s for one epoch
    train_data = load_data("road_data/train", transform_pipeline="default", shuffle=True, batch_size=batch_size, num_workers=2)
    train_data_aug = load_data("road_data/train", transform_pipeline="aug", shuffle=True, batch_size=batch_size, num_workers=2)
    val_data = load_data("road_data/val", shuffle=False)

    # create loss function and optimizer
    segmentation_loss_func = ClassificationLoss()
    regression_loss_func = RegressionLoss()
    # optimizer = ...
    optim = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    global_step = 0
    metric = DetectionMetric()

    # training loop
    for epoch in range(num_epoch):
        # clear metrics at beginning of epoch
        metric.reset()

        model.train()

        print("train data default")
        start_time = time.time()
        for data in train_data:
            image = data["image"]
            depth = data["depth"]
            track = data["track"]
            image, depth, track = image.to(device), depth.to(device), track.to(device)

            pred_track, pred_depth = model(image)

            loss_val_track = segmentation_loss_func(pred_track, track)
            loss_val_depth = regression_loss_func(pred_depth, depth)
            combined_loss = loss_val_track + loss_val_depth
            
            optim.zero_grad()
            combined_loss.backward()
            optim.step()

            global_step += 1
            pred_track_class = pred_track.argmax(dim=1)
            metric.add(pred_track_class, track, pred_depth, depth)
            logger.add_scalar('train_loss_track', loss_val_track.item(), global_step)
            logger.add_scalar('train_loss_depth', loss_val_depth.item(), global_step)
        end_time = time.time()
        print(f"Elapsed time: {end_time - start_time} seconds")

        print("train data aug")
        for data in train_data_aug:
            image = data["image"]
            depth = data["depth"]
            track = data["track"]
            image, depth, track = image.to(device), depth.to(device), track.to(device)

            pred_track, pred_depth = model(image)

            loss_val_track = segmentation_loss_func(pred_track, track)
            loss_val_depth = regression_loss_func(pred_depth, depth)
            combined_loss = loss_val_track + loss_val_depth
            
            optim.zero_grad()
            combined_loss.backward()
            optim.step()

            global_step += 1
            pred_track_class = pred_track.argmax(dim=1)
            metric.add(pred_track_class, track, pred_depth, depth)
            logger.add_scalar('train_loss_track', loss_val_track.item(), global_step)
            logger.add_scalar('train_loss_depth', loss_val_depth.item(), global_step)
        
        training_accurancy = metric.compute()
        logger.add_scalar('train_track_accuracy', training_accurancy["accuracy"], epoch)
        logger.add_scalar('train_depth_error', training_accurancy["abs_depth_error"], epoch)
        
        # disable gradient computation and switch to evaluation mode
        with torch.inference_mode():
            model.eval()
            metric.reset()
            print("val data")
            for data in val_data:
                image = data["image"]
                depth = data["depth"]
                track = data["track"]
                image, depth, track = image.to(device), depth.to(device), track.to(device)

                pred_track, pred_depth = model(image)
                
                # Need to take max on last dimension
                pred_track_class = pred_track.argmax(dim=1)
                metric.add(pred_track_class, track, pred_depth, depth)
                

        val_accurancy = metric.compute()
        logger.add_scalar('val_track_accuracy', val_accurancy["accuracy"], epoch)
        logger.add_scalar('val_depth_error', val_accurancy["abs_depth_error"], epoch)

        # print on first, last, every 10th epoch
        # if epoch == 0 or epoch == num_epoch - 1 or (epoch + 1) % 10 == 0:
        print(
            f"Epoch {epoch + 1:2d} / {num_epoch:2d}: "
            f"train_track_acc={training_accurancy["accuracy"]:.4f} "
            f"train_depth_err={training_accurancy["abs_depth_error"]:.4f} "
            f"val_track_acc={val_accurancy["accuracy"]:.4f} "
            f"val_depth_err={val_accurancy["abs_depth_error"]:.4f}"
        )

    # save and overwrite the model in the root directory for grading
    save_model(model)

    # save a copy of model weights in the log directory
    torch.save(model.state_dict(), log_dir / f"{model_name}.th")
    print(f"Model saved to {log_dir / f'{model_name}.th'}")


if __name__ == "__main__":
    train()
