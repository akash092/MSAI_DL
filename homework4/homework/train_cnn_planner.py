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
    model_name: str = "cnn_planner",
    num_epoch: int = 20,
    lr: float = 1e-2,
    seed: int = 2024,
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

    model = load_model(model_name)
    model = model.to(device)
    model.train()

    train_data = load_data("drive_data/train", shuffle = True, num_workers=0, batch_size=64)
    val_data = load_data("drive_data/val", shuffle=False, num_workers=0)

    # create loss function and optimizer
    
    # create loss function and optimizer
    myloss_fn = MLPPlannerLoss()
    
    # logging purposes only
    metric = PlannerMetric()
    metric_global = PlannerMetric()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)



    # training loop
    min_validation_err = 10
    for epoch in range(num_epoch):
        model.train()
        metric_global.reset()

        for data in train_data:

            image = data["image"].to(device)
            waypoints, waypoints_mask = data["waypoints"].to(device), data["waypoints_mask"].to(device)
            pred = model(image)
            
            loss_train_comb = myloss_fn(pred, waypoints, waypoints_mask)
            loss_train = loss_train_comb["l1_error"]
            
            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()

            metric_global.add(pred, waypoints, waypoints_mask)

        # disable gradient computation and switch to evaluation mode
        with torch.inference_mode():
            model.eval()
            metric.reset()
            for data in val_data:
                image = data["image"].to(device)
                waypoints, waypoints_mask = data["waypoints"].to(device), data["waypoints_mask"].to(device)
                pred = model(image)
                metric.add(pred, waypoints, waypoints_mask)
                
            loss_val_comb = metric.compute()

        # log training error for this epoch
        train_loss = metric_global.compute()

        print(
            f"Epoch {epoch + 1:2d} / {num_epoch:2d}: "
            f"train_error={train_loss["l1_error"]:.4f} "
            f"val_error={loss_val_comb["l1_error"]:.4f}"
        )

        if loss_val_comb["l1_error"] < min_validation_err:
            min_validation_err = loss_val_comb["l1_error"]
            save_model_epoch(model, str(epoch))


    # # save and overwrite the model in the root directory for grading
    # save_model(model)


if __name__ == "__main__":
    train()

