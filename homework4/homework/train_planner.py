"""
Usage:
    python3 -m homework.train_planner --your_args here
"""

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.utils.tensorboard as tb
import torch.optim as optim
from .models import load_model, save_model, save_model_epoch, MLPPlannerLoss
from .datasets.road_dataset import load_data
from .metrics import PlannerMetric


def train(
    exp_dir: str = "logs",
    model_name: str = "mlp_planner",
    num_epoch: int = 50,
    lr: float = 1e-3,
    seed: int = 2024,
    **kwargs,
):
    print(f"{model_name=} {num_epoch=} {lr=}")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        print("CUDA not available, using CPU")
        device = torch.device("cpu")

    # set random seed so each run is deterministic
    torch.manual_seed(seed)
    np.random.seed(seed)

    # directory with timestamp to save tensorboard logs and model checkpoints
    log_dir = Path(exp_dir) / f"{model_name}_{datetime.now().strftime('%m%d_%H%M%S')}"
    logger = tb.SummaryWriter(log_dir)

    model = load_model(model_name, **kwargs)
    model = model.to(device)
    model.train()

    train_data = load_data("drive_data/train", "state_only", shuffle = True, num_workers=0, batch_size=128)
    val_data = load_data("drive_data/val", "state_only", shuffle=False, num_workers=0)

    # create loss function and optimizer
    
    myloss_fn = MLPPlannerLoss()
    
    # logging purposes only
    metric = PlannerMetric()
    metric_global = PlannerMetric()

    # optimizer = ...
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)


    global_step = 0

    # training loop
    min_validation_err = 10
    for epoch in range(num_epoch):
        model.train()
        metric_global.reset()

        for data in train_data:

            track_left, track_right = data["track_left"].to(device), data["track_right"].to(device)
            waypoints, waypoints_mask = data["waypoints"].to(device), data["waypoints_mask"].to(device)
            pred = model(track_left, track_right)
            
            loss_train_comb = myloss_fn(pred, waypoints, waypoints_mask)
            loss_train = loss_train_comb["l1_error"]
            
            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()

            global_step += 1
            logger.add_scalar('train_error', loss_train, global_step)
            logger.add_scalar('train_long_error', loss_train_comb["longitudinal_error"], global_step)
            logger.add_scalar('train_lat_error', loss_train_comb["lateral_error"], global_step)

            metric_global.add(pred, waypoints, waypoints_mask)

        # disable gradient computation and switch to evaluation mode
        with torch.inference_mode():
            model.eval()
            metric.reset()
            for data in val_data:
                track_left, track_right = data["track_left"].to(device), data["track_right"].to(device)
                waypoints, waypoints_mask = data["waypoints"].to(device), data["waypoints_mask"].to(device)
                pred = model(track_left, track_right)
                metric.add(pred, waypoints, waypoints_mask)
                
            loss_val_comb = metric.compute()
            # log validation error for this epoch
            logger.add_scalar('val_error', loss_val_comb["l1_error"], epoch)
            logger.add_scalar('val_long_error', loss_val_comb["longitudinal_error"], epoch)
            logger.add_scalar('val_lat_error', loss_val_comb["lateral_error"], epoch)

        # log training error for this epoch
        train_loss = metric_global.compute()

        print(
            f"Epoch {epoch + 1:2d} / {num_epoch:2d}: "
            f"train_error={train_loss["l1_error"]:.4f} "
            f"val_error={loss_val_comb["l1_error"]:.4f}"
        )
        scheduler.step()  # Update learning rate at the end of each epoch
        if loss_val_comb["l1_error"] < min_validation_err:
            min_validation_err = loss_val_comb["l1_error"]
            save_model_epoch(model, str(epoch))


    # # save and overwrite the model in the root directory for grading
    # save_model(model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--exp_dir", type=str, default="logs")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--num_epoch", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=2024)

    # optional: additional model hyperparamters
    # parser.add_argument("--num_layers", type=int, default=5)
    # parser.add_argument("--size_hidden", type=int, default=128)

    # pass all arguments to train
    train(**vars(parser.parse_args()))

