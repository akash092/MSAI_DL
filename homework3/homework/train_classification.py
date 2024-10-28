from datetime import datetime
from pathlib import Path

import torch
import torch.utils.tensorboard as tb

from .models import load_model, save_model
from .metrics import AccuracyMetric
from .datasets.classification_dataset import load_data


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
    
def train(
    exp_dir: str = "logs",
    model_name: str = "classifier",
    num_epoch: int = 10,
    lr: float = 1e-2,
    batch_size: int = 128,
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
    model.train() # inidcates to tensor that we are training right . some behavior is different compared to eval()

    train_data_1 = load_data("classification_data/train", transform_pipeline="default", shuffle=True, batch_size=batch_size, num_workers=0)
    train_data_2 = load_data("classification_data/train", transform_pipeline="aug", shuffle=True, batch_size=batch_size, num_workers=0)

    val_data = load_data("classification_data/val", shuffle=False)

    # create loss function and optimizer
    loss_func = ClassificationLoss()
    # optimizer = ...
    optim = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    global_step = 0
    metric = AccuracyMetric()

    # training loop
    for epoch in range(num_epoch):
        # clear metrics at beginning of epoch
        metric.reset()

        model.train()

        for img, label in train_data_1:
            img, label = img.to(device), label.to(device)

            pred = model(img) # pred will be of size batch_size X class_size
            loss_val = loss_func(pred, label)

            optim.zero_grad()
            loss_val.backward()
            optim.step()

            global_step += 1
            pred_class = pred.argmax(dim=1)
            metric.add(pred_class, label)
            logger.add_scalar('train_loss', loss_val.item(), global_step)
        
        for img, label in train_data_2:
            img, label = img.to(device), label.to(device)

            pred = model(img) # pred will be of size batch_size X class_size
            loss_val = loss_func(pred, label)

            optim.zero_grad()
            loss_val.backward()
            optim.step()

            global_step += 1
            pred_class = pred.argmax(dim=1)
            metric.add(pred_class, label)
            logger.add_scalar('train_loss', loss_val.item(), global_step)

        training_accurancy = metric.compute()
        epoch_train_acc = training_accurancy["accuracy"]
        logger.add_scalar('train_accuracy', epoch_train_acc, epoch)
        
        # disable gradient computation and switch to evaluation mode
        with torch.inference_mode():
            model.eval()
            metric.reset()
            for img, label in val_data:
                img, label = img.to(device), label.to(device)

                pred = model(img)
                pred_class = pred.argmax(dim=1)
                metric.add(pred_class, label)
                

        val_accurancy = metric.compute()
        epoch_val_acc = val_accurancy["accuracy"]
        logger.add_scalar('val_accuracy', epoch_val_acc, epoch)

        # print on first, last, every 10th epoch
        # if epoch == 0 or epoch == num_epoch - 1 or (epoch + 1) % 10 == 0:
        print(
            f"Epoch {epoch + 1:2d} / {num_epoch:2d}: "
            f"train_acc={epoch_train_acc:.4f} "
            f"val_acc={epoch_val_acc:.4f}"
        )

    # save and overwrite the model in the root directory for grading
    save_model(model)

    # save a copy of model weights in the log directory
    torch.save(model.state_dict(), log_dir / f"{model_name}.th")
    print(f"Model saved to {log_dir / f'{model_name}.th'}")


if __name__ == "__main__":
    train()
