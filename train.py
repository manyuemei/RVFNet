import os
import argparse
import matplotlib.pyplot as plt
import torch
import torch.optim as optim

from torchvision import transforms
from makeset import VideoDataset
from utils import train_one_epoch, evaluate
from model.spatial import Spatialpath
from torch.cuda.amp import autocast, GradScaler

def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    scaler = GradScaler()

    if os.path.exists("./checkpoints") is False:
        os.makedirs("./checkpoints")

    root_folder_train = r"./dataset/train"
    train_path_label = r"./label/train_label.txt"
    root_folder_val = r"./dataset/val"
    val_path_label = r"./label/val_label.txt"

    img_size = 224

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(img_size),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]),
        "val": transforms.Compose([transforms.Resize(int(img_size * 1.143)),
                                   transforms.CenterCrop(img_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])}

    train_datasets = VideoDataset(root_folder_train, train_path_label, transform=data_transform["train"])

    val_datasets = VideoDataset(root_folder_val, val_path_label, transform=data_transform["val"])

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 4])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_datasets,
                                               batch_size=batch_size,
                                               shuffle=False,
                                               pin_memory=False,
                                               num_workers=nw,
                                               )

    val_loader = torch.utils.data.DataLoader(val_datasets,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=False,
                                             num_workers=nw,
                                             )

    model = Spatialpath(hash_bits=args.hash_bits).to(device)

    if args.freeze_layers:
        for name, para in model.named_parameters():
            if "fc" not in name:
                para.requires_grad_(False)
            else:
                print("training {}".format(name))

    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(pg, lr=args.lr, weight_decay=2E-5, eps=1E-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 5, gamma=0.5)

    if args.weights != "":
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
        checkpoint = torch.load(args.weights)
        weights_dict = checkpoint['model_state_dict']
        model.load_state_dict(weights_dict, strict=False)
        #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epochx = checkpoint['epoch']
        print('load epoch',epochx)

    train_loss1 = []
    loss_path = './train_loss' + '_' + str(args.hash_bits) + 'bits_' + str(args.margin) + 'margin_' + str(
        args.lr) + 'lr'
    loss_path1 = './val_loss' + '_' + str(args.hash_bits) + 'bits_' + str(args.margin) + 'margin_' + str(args.lr) + 'lr'
    imgPath = './losspic/' + '_' + str(args.hash_bits) + 'bits_' + str(args.margin) + 'margin_' + str(args.lr) + 'lr'
    checkpoint_path = './checkpoints/' + '_' + str(args.hash_bits) + 'bits_' + str(args.margin) + 'margin_' + str(
        args.lr) + 'lr'
    hashdistance_path = './hash_distance/' + str(args.hash_bits) + 'bits_' + str(args.margin) + 'margin_' + str(
        args.lr) + 'lr'

    os.makedirs(checkpoint_path, exist_ok=True)
    os.makedirs(loss_path, exist_ok=True)
    os.makedirs(loss_path1, exist_ok=True)
    os.makedirs(hashdistance_path, exist_ok=True)
    os.makedirs(imgPath, exist_ok=True)

    for epoch in range(0, args.epochs):
        # train
        train_loss, elapsed_time = train_one_epoch(model=model,
                                                   optimizer=optimizer,
                                                   data_loader=train_loader,
                                                   device=device,
                                                   epoch=epoch,
                                                   args=args,
                                                   scheduler=scheduler,
                                                   scaler=scaler)
        train_loss1.append(train_loss)

        f = open(os.path.join(loss_path, "loss.log"), "a+")
        f.write('epoch:' + str(epoch) + "  loss:" + str(train_loss) +"  time/s:" + str(elapsed_time) + '\n')
        print(f'[{epoch + 1}/{args.epochs}] loss:{train_loss:0.5f} '
              f' time:{elapsed_time:0.2f} s')

        path_checkpoint = os.path.join(checkpoint_path, 'checkpoint_{}_epoch.pkl'.format(epoch))
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, path_checkpoint)

        # validate
        val_loss, elapsed_time1 = evaluate(model=model,
                                           data_loader=val_loader,
                                           device=device,
                                           epoch=epoch,
                                           args=args,
                                           path=hashdistance_path)

        f = open(os.path.join(loss_path1, "loss.log"), "a+")
        f.write('epoch:' + str(epoch) + "  loss:" + str(val_loss) + "  time/s:" + str(elapsed_time) + '\n')
        print(f'[{epoch + 1}/{args.epochs}] loss:{val_loss:0.5f} '
              f' time:{elapsed_time1:0.2f} s')

    plt.switch_backend('Agg')
    plt.figure()
    plt.plot(train_loss1, color='r', linestyle='--', label='train_loss')
    plt.ylabel('train_loss')
    plt.xlabel('epoches')
    plt.legend()
    plt.savefig(os.path.join(imgPath, str(args.epochs) + "epoch_train_loss.jpg"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hash_bits', type=int, default=64, help='length of fingerprint')
    parser.add_argument('--margin', type=float, default='18')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=69)
    parser.add_argument('--lr', type=float, default=0.003)
    parser.add_argument('--weights', type=str, default='',
                        help='initial weights path')
    parser.add_argument('--freeze_layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    main(opt)
