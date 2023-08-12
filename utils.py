import sys
import time
import numpy as np
import torch

from hard_triplet import TripletLoss
from torch.cuda.amp import autocast


def train_one_epoch(model, optimizer, data_loader, device, epoch, args, scheduler):
    start_time = time.time()
    model.train()
    triplet_loss = TripletLoss(margin=args.margin).to(device)
    accu_loss = 0.0

    for step, data in enumerate(data_loader, start=0):
        print('\n-----------Training------------  epoch, step:', epoch, step)
        optimizer.zero_grad()
        images, labels = data

        # AMP
        with torch.autograd.set_detect_anomaly(True):
            with autocast():
                _, _, final = model(images.to(device))
                loss = triplet_loss(final, labels.to(device))

            loss.backward()

        accu_loss += loss.detach().item()

        rate = (step + 1) / len(data_loader)
        a = "*" * int(rate * 50)
        b = "." * int((1 - rate) * 50)
        print("\rtrain loss: {:^3.0f}%[{}->{}]{:.4f}".format(int(rate * 100), a, b, loss), end="")

        v_n =[]
        v_v =[]
        v_g=[]
        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            for name,parameter in model.named_parameters():
                v_n.append(name)
                v_v.append(parameter.detach().cpu().numpy() if parameter is not None else [0])
                v_g.append(parameter.grad.detach().cpu().numpy() if parameter is not None else [0])
            for j in range(len(v_n)):
                print('-----',v_n[j],'------',np.min(v_v[j]).item(),np.max(v_v[j]).item())
                print('-----', v_n[j], '------', np.min(v_g[j]).item(), np.max(v_g[j]).item())
            sys.exit(1)

        optimizer.step()

        del images, labels, final, loss
        torch.cuda.empty_cache()

    scheduler.step()

    avg_loss = accu_loss / (step+1)

    end_time = time.time()
    elapsed_time = end_time - start_time

    print('\n------------Finish training------------  epoch:', epoch)

    return avg_loss, elapsed_time


@torch.no_grad()
def evaluate(model, data_loader, device, epoch, args, path):
    triplet_loss1 = TripletLoss(margin=args.margin, epoch=epoch, path=path, val=True).to(device)
    model.eval()
    start_time = time.time()
    accu_loss = 0.0

    for step, data in enumerate(data_loader):
        print('-------------Validation--------------  epoch, step:', epoch, step)
        images, labels = data

        with autocast():
            _, _, final = model(images.to(device))
            loss = triplet_loss1(final, labels.to(device))

        rate = (step + 1) / len(data_loader)
        a = "*" * int(rate * 50)
        b = "." * int((1 - rate) * 50)
        print("\rval loss: {:^3.0f}%[{}->{}]{:.4f}".format(int(rate * 100), a, b, loss), end="")
        accu_loss += loss.item()

        del images, labels, final, loss
        torch.cuda.empty_cache()

    end_time = time.time()
    elapsed_time = end_time - start_time

    print('\n-------------Finish validation--------------  epoch:', epoch)

    return accu_loss / (step+1), elapsed_time
