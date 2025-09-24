from model.decoderstyle import DecoderStyle, DecodeStyleParams
from data.decoderstyle import VillahuDecoderStyle, collate_fn
from utils import diff_words

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import logging
import os
from dataclasses import dataclass, asdict
from datetime import datetime
import argparse


@dataclass
class TrainParams:
    df_path:str
    preprop_fn:str
    batch_size:int
    epochs:int
    lr:float
    device:str
    log_every_nsteps:int
    save_every_nsteps:int
    save_dir:str = "checkpoints"

def train(model_args:DecodeStyleParams, train_args:TrainParams):
    model = DecoderStyle(model_args).to(train_args.device)

    preprop_fn = getattr(diff_words, train_args.preprop_fn)
    dataset = VillahuDecoderStyle(train_args.df_path, preprop=preprop_fn)
    dataloader = DataLoader(dataset, batch_size=train_args.batch_size, shuffle=True, collate_fn=collate_fn)

    optimizer = torch.optim.Adam(model.parameters(), lr=train_args.lr)
    criteria = torch.nn.CrossEntropyLoss().to(train_args.device)

    for epoch in range(train_args.epochs):
        for bidx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            model.zero_grad()

            texts, label = batch
            print(texts)
            print("--------------")
            print(label)
            print("--------------")
            label = label.to(train_args.device)
            logits = model(texts)
            loss = criteria(logits, label)

            loss.backward()
            optimizer.step()

            if bidx % train_args.log_every_nsteps == 0:
                logging.info(f"Epoch {epoch}, Step {bidx}, Loss {loss.item():.4f}")
            if bidx % train_args.save_every_nsteps == 0:
                save_checkpoint(epoch, model, optimizer, None, loss, train_args)


def save_checkpoint(epoch, model, optimizer, scheduler, loss, config):
    os.makedirs(config.save_dir, exist_ok=True)
    ckpt_path = os.path.join(config.save_dir, f"checkpoint_epoch{epoch}.pt")
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler else None,
        "loss": loss.item(),
        "config": asdict(config)
    }, ckpt_path)
    logging.info(f"Checkpoint saved: {ckpt_path}")

def setup_logger():
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = os.path.join("logs", f"traintorch_testsmall_{current_time}.txt")
    logging.basicConfig(filename=filename, filemode="w", level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s")
    logging.info("Logger Initialized")
    return filename

def log_configs(model_cfg, train_cfg):
    logging.info("===== Training Configuration =====")
    for k, v in asdict(train_cfg).items():
        logging.info(f"{k}: {v}")
    logging.info("===== Model Dimensions =====")
    for k, v in asdict(model_cfg).items():
        logging.info(f"{k}: {v}")

def setup():
    parser = argparse.ArgumentParser(description="My training script")

    parser.add_argument("--llm_name", type=str, default="bert-base-uncased")
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--num_atnns", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--df_path", type=str, required=True)
    parser.add_argument("--preprop_fn", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--log_every_nsteps", type=int, default=10)
    parser.add_argument("--save_every_nsteps", type=int, default=100)
    
    return parser


def get_config(parser:argparse.ArgumentParser):
    args = parser.parse_args()

    model_args = DecodeStyleParams(
        llm_name=args.llm_name,
        num_heads=args.num_heads,
        num_atnns=args.num_atnns,
        dropout=args.dropout,
        huggingface_device=args.device,
    )

    train_args = TrainParams(
        df_path=args.df_path,
        preprop_fn=args.preprop_fn,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        device=args.device,
        log_every_nsteps=args.log_every_nsteps,
        save_every_nsteps=args.save_every_nsteps,
    )

    return model_args, train_args

if __name__ == "__main__":
    parser = setup()
    model_cfg, train_cfg = get_config(parser)
    log_file = setup_logger()
    log_configs(model_cfg, train_cfg)
    logging.info(f"Logging to {log_file}")
    train(model_cfg, train_cfg)
