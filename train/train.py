"""
script to train the models
"""

import os

import random
import time
import sys

from tqdm import tqdm
import wandb


import torch
from torch import nn as nn
from torch.utils.data import DataLoader

from torchvision import transforms

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from models.model import CnnModel
from datasets.dataset import CatDogDataset

EPOCHS = 20
CPU_WORKERS: int = os.cpu_count() or 1

loss_fn = nn.CrossEntropyLoss()


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_files = os.listdir("data/train/train")
    img_files = list(filter(lambda x: x != "train", img_files))

    def train_path(p: str) -> str:
        return f"data/train/train/{p}"

    img_files = list(map(train_path, img_files))

    random.shuffle(img_files)
    train = img_files[:20000]
    test = img_files[20000:]

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    train_ds = CatDogDataset(train, transform)
    # train_dl = DataLoader(train_ds, 100)

    test_ds = CatDogDataset(test, transform)
    # test_dl = DataLoader(test_ds, 100)

    train_dl = DataLoader(train_ds, batch_size=128, num_workers=CPU_WORKERS, pin_memory=True, prefetch_factor=2, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=128, num_workers=CPU_WORKERS, pin_memory=True)

    # Create instance of the model
    model = CnnModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.to(device)
    # model = model.compile()

    losses = []
    accuracies = []
    start = time.time()
    loss_fn = nn.CrossEntropyLoss()

    config = {
        "learning_rate": 0.001,
        "architecture": "CNN",
        "dataset": "dogs-vs-cats-redux-kernels-edition",
        "epochs": 8,
    }
    with wandb.init(project="cat-vs-dog", config=config) as run:
        # Model Training...
        for epoch in range(EPOCHS):
            epoch_loss = 0
            epoch_accuracy = 0

            for X, y in tqdm(train_dl):
                # Move data to GPU
                X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
                preds = model(X)
                loss = loss_fn(preds, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                accuracy = (preds.argmax(dim=1) == y).float().mean()
                epoch_accuracy += accuracy
                epoch_loss += loss
                # print('.', end='', flush=True)

            epoch_accuracy = epoch_accuracy / len(train_dl)
            accuracies.append(epoch_accuracy)
            epoch_loss = epoch_loss / len(train_dl)
            losses.append(epoch_loss)

            # print("\n --- Epoch: {}, train loss: {:.4f}, train acc: {:.4f}, time: {}".format(epoch, epoch_loss, epoch_accuracy, time.time() - start))
            run.log({"epoch": epoch, "loss": epoch_loss, "accuracy": epoch_accuracy})

            # test set accuracy
            with torch.no_grad():
                test_epoch_loss = 0
                test_epoch_accuracy = 0

                for test_X, test_y in test_dl:
                    test_X, test_y = (
                        test_X.to(device, non_blocking=True),
                        test_y.to(device, non_blocking=True),
                    )

                    test_preds = model(test_X)
                    test_loss = loss_fn(test_preds, test_y)

                    test_epoch_loss += test_loss
                    test_accuracy = (test_preds.argmax(dim=1) == test_y).float().mean()
                    test_epoch_accuracy += test_accuracy

                test_epoch_accuracy = test_epoch_accuracy / len(test_dl)
                test_epoch_loss = test_epoch_loss / len(test_dl)

                # print("Epoch: {}, test loss: {:.4f}, test acc: {:.4f}, time: {}\n".format(epoch, test_epoch_loss, test_epoch_accuracy, time.time() - start))
                run.log(
                    {
                        "test epoch": epoch,
                        "test loss": test_epoch_loss,
                        "test accuracy": test_epoch_accuracy,
                    }
                )

    # torch.save(model.state_dict(), "cat_dog_model_weights.pth")
    model.save_pretrained("cnn_cat_vs_dog")
    upload()

def upload():
    model = CnnModel.from_pretrained("cnn_cat_vs_dog")
    print("model loaded")
    model.push_to_hub(repo_id="sachinkum0009/cnn_cat_vs_dog", branch="dev")


if __name__ == "__main__":
    train()
