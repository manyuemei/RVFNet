import os
import argparse
import torch
from torchvision import transforms
import time
import csv

from makeset import VideoDataset
from model.spatial import Spatialpath


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    root_folder_test = r"./dataset/test"
    test_path_label = r"./label/test_label.txt"

    img_size = 224

    data_transform = {
        "test": transforms.Compose([transforms.Resize(int(img_size)),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])}


    test_datasets = VideoDataset(root_folder_test, test_path_label, transform=data_transform["test"])

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 4])
    print('Using {} dataloader workers every process'.format(nw))
    test_loader = torch.utils.data.DataLoader(test_datasets,
                                               batch_size=batch_size,
                                               shuffle=False,
                                               pin_memory=False,
                                               num_workers=nw,
                                               )

    model = Spatialpath(hash_bits=args.hash_bits).to(device)

    for name, para in model.named_parameters():
        para.requires_grad_(False)

    if args.weights != "":
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
        checkpoint = torch.load(args.weights)
        weights_dict = checkpoint['model_state_dict']
        model.load_state_dict(weights_dict, strict=False)
        epochx = checkpoint['epoch']
        print('load epoch',epochx)

    hashcode_path = './test_hash_distance/' + str(args.hash_bits) + 'bits'
    time_path = './test_time/'

    os.makedirs(hashcode_path, exist_ok=True)
    os.makedirs(time_path, exist_ok=True)

    model.eval()
    start_time = time.time()
    for step, data in enumerate(test_loader):
        images, _ = data

        with torch.no_grad():
            _, _, final = model(images.to(device))

        final = final.squeeze()

        with open(os.path.join(hashcode_path, "fingerprint.csv"), 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            row_to_write = ['{:.5f}'.format(single_code.item()) for single_code in final]
            print(row_to_write)
            writer.writerow(row_to_write)

        del images, final
        torch.cuda.empty_cache()

    end_time = time.time()
    elapsed_time = end_time - start_time
    aver_time = elapsed_time/(step+1)
    print('Cost {:.2f} sec'.format(elapsed_time))
    f = open(os.path.join(time_path, "time.log"), "a+")  # a+可读可写
    f.write('model:64_RVFNet' + "  average_time/s:" + str(aver_time) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hash_bits', type=int, default=64, help='length of hashing binary')
    parser.add_argument('--batch_size', type=int, default=1)

    parser.add_argument('--weights', type=str, default='',
                        help='initial weights path')

    parser.add_argument('--device', default='cuda:1', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    main(opt)
