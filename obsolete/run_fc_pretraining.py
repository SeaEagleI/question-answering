# -*- coding: utf-8 -*-
import os
import time
import torch
from torch import optim, nn
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
from transformers import AlbertConfig
from models import FCLayer
from squad_v2 import SquadDataset
from pytorchtools import EarlyStopping

# file paths
data_dir = 'squad_v2'
dataset_dir = os.path.join(data_dir, 'embeddings')
weights_dir = os.path.join(data_dir, 'weights')
model_path_tplt = os.path.join(weights_dir, 'fc-{}-{:.4f}.pth')
best_model_path = os.path.join(weights_dir, 'fc_best_model.pth')
preload_model_path = os.path.join(weights_dir, 'ori_model.pth')
model_name_or_path = './albert-xxlarge-v2/finetuned-ckpt-44066'
# model hyper-paras
max_seq_length = 384
batch_size = 8 * 4
save_interval = 3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_size = 15000
eval_size = 5000
learning_rate = 1e-3
pretrained = True
n_gpu = torch.cuda.device_count()
epochs = 50


# Train steps
def train(model, train_dataloader, epoch_id, loss_fct, optimizer):
    model.train()
    print("Epoch {}/{}".format(epoch_id, epochs))
    total_loss, total_correct, total, batches = 0.0, 0, 0, len(train_dataloader)
    start_t = time.time()
    for step_id, batch in enumerate(train_dataloader):
        batch = tuple(t.to(device) for t in batch)
        inputs = {
            "albert_outputs": batch[0],  # shape: [batch_size, max_seq_len, hidden_dim]
        }
        start_positions = batch[3]
        end_positions = batch[4]
        # get outputs
        start_logits, end_logits = model(**inputs)
        start_logits = start_logits.squeeze(-1)  # shape: [batch_size, max_seq_len]
        end_logits = end_logits.squeeze(-1)  # shape: [batch_size, max_seq_len]

        # If we are on multi-GPU, split add a dimension
        if len(start_positions.size()) > 1:
            start_positions = start_positions.squeeze(-1)  # shape: [batch_size]
        if len(end_positions.size()) > 1:
            end_positions = end_positions.squeeze(-1)  # shape: [batch_size]
        # sometimes the start/end positions are outside our model inputs, we ignore these terms
        start_positions.clamp_(0, max_seq_length)
        end_positions.clamp_(0, max_seq_length)

        cur_batch_size = start_positions.size(0)
        total += cur_batch_size
        # compute train_loss: CELoss([batch_size, max_seq_len], [batch_size]) => loss (scalar, [])
        start_loss = loss_fct(start_logits, start_positions)
        end_loss = loss_fct(end_logits, end_positions)
        loss = (start_loss + end_loss) / 2
        if n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel (not distributed) training
        total_loss += loss.item() * cur_batch_size
        # back propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        # compute train_em
        preds = torch.cat([start_logits.argmax(1).unsqueeze(1), end_logits.argmax(1).unsqueeze(1)], dim=1)
        gts = torch.cat([start_positions.unsqueeze(1), end_positions.unsqueeze(1)], dim=1)
        correct = sum([list(preds[i]) == list(gts[i]) for i in range(cur_batch_size)])
        total_correct += correct
        # log train_loss, train_em
        used_t = time.time() - start_t
        rest_t = used_t / (step_id + 1) * (batches - step_id - 1)
        print("\rsteps: {}/{}\t{:.1f}%\ttime:{:.2f}min\teta:{:.2f}min\tcur_em: {:.4f}\ttrain_avg_em: {:.4f}\t"
              "train_loss: {:.4f}".format(step_id + 1, batches, (step_id+1)*100/batches, used_t / 60, rest_t / 60,
                                          correct / cur_batch_size, total_correct / total, total_loss / total), end='')
    print()
    # return
    return model, optimizer


# Evaluate steps
def evaluate(model, eval_dataloader, epoch_id, loss_fct):
    model.eval()
    total_loss, total_correct, total = 0.0, 0, 0
    with torch.no_grad():
        for step_id, batch in enumerate(tqdm(eval_dataloader, desc="Evaluating")):
            batch = tuple(t.to(device) for t in batch)
            inputs = {
                "albert_outputs": batch[0],  # shape: [batch_size, max_seq_len, hidden_dim]
            }
            start_positions = batch[3]
            end_positions = batch[4]
            # get outputs
            start_logits, end_logits = model(**inputs)
            start_logits = start_logits.squeeze(-1)  # shape: [batch_size, max_seq_len]
            end_logits = end_logits.squeeze(-1)  # shape: [batch_size, max_seq_len]

            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)  # shape: [batch_size]
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)  # shape: [batch_size]
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            start_positions.clamp_(0, max_seq_length)
            end_positions.clamp_(0, max_seq_length)

            cur_batch_size = start_positions.size(0)
            total += cur_batch_size
            # compute val_loss
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            loss = (start_loss + end_loss) / 2
            total_loss += loss.item() * cur_batch_size
            # compute val_em
            preds = torch.cat([start_logits.argmax(1).unsqueeze(1), end_logits.argmax(1).unsqueeze(1)], dim=1)
            gts = torch.cat([start_positions.unsqueeze(1), end_positions.unsqueeze(1)], dim=1)
            total_correct += sum([list(preds[i]) == list(gts[i]) for i in range(cur_batch_size)])
        # log val_loss, val_em
        val_loss = total_loss / total
        val_em = total_correct / total
        print("val_loss: {:.4f}\tval_em: {:.4f}".format(val_loss, val_em))
    # return
    return model, val_em


def main():
    # Setup model
    config = AlbertConfig.from_pretrained(model_name_or_path)
    model = FCLayer(config.hidden_size, config.num_labels)   # fc layer comparison with ptr-net
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.nn.DataParallel(model).to(device)
    if pretrained and os.path.exists(preload_model_path):
        model.load_state_dict(torch.load(preload_model_path))
        print('loaded model from {}'.format(preload_model_path))
    # create weights_dir
    if not os.path.exists(weights_dir):
        os.makedirs(weights_dir)

    # Generate data_loader
    train_dataset = SquadDataset(dataset_dir, 'train', train_size)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    eval_dataset = SquadDataset(dataset_dir, 'val', eval_size)
    eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

    # Define Loss Function, Optimizer & ReduceLROnPlateau, EarlyStopping
    loss_fct = nn.CrossEntropyLoss(ignore_index=max_seq_length).to(device)
    # optimizer = AdamW(model.parameters(), lr=5e-5, eps=1e-8)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3,
                                                     verbose=True, threshold=1e-4)
    early_stopping = EarlyStopping(patience=5, verbose=True, path=best_model_path)

    # Start training!
    model.zero_grad()
    for epoch_id in trange(1, epochs + 1, desc="Epoch"):
        # Train
        # model, optimizer = train(model, train_dataloader, epoch_id, loss_fct, optimizer)
        # Evaluate at the end of each epoch
        model, val_em = evaluate(model, eval_dataloader, epoch_id, loss_fct)
        # Save model checkpoint when epoch ends, or save_steps arrives
        if epoch_id % save_interval == 0 or epoch_id == epochs:
            torch.save(model.state_dict(), model_path_tplt.format(epoch_id, val_em))
        # Early Stopping
        early_stopping(val_em, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break
        # LR Decay
        scheduler.step(val_em)  # Update learning rate schedule


if __name__ == '__main__':
    main()
