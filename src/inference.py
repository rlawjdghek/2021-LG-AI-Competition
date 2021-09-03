from torch.utils.data import Dataset, DataLoader
import torch
import glob
import os
import numpy as np
import cv2
from multiprocessing import Pool, Manager, Process
import torchvision.transforms as transforms
import argparse
import segmentation_models_pytorch as smp
import torch.cuda.amp as amp
import torch.nn as nn
import time
import matplotlib.pyplot as plt
########
final_start_time = time.time()
########
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
is_print = True
patch_size = 768
overlap_ratio = 0.5
max_y = 7
max_x = 9
manager = Manager()
# data_dict = manager.dict()

def crop_image(filename):
    overlap = 0.5
    dict = {}
    flag = False
    for origin_filename in filename:
        origin_image = cv2.imread(origin_filename)
        filename_only = origin_filename.split("/")[-1].split(".")[0]
        x_list = [i for i in range(0, len(origin_image[0]), int(patch_size * (1 - overlap)))]
        y_list = [i for i in range(0, len(origin_image), int(patch_size * (1 - overlap)))]
        for idx_x, x in enumerate(x_list):
            for idx_y, y in enumerate(y_list):
                patch_image = origin_image[y:y+patch_size, x:x+patch_size, :]
                if patch_image.shape != (patch_size, patch_size, 3):
                    h, w, c  = patch_image.shape
                    if not flag:
                        zeros = np.zeros((patch_size, patch_size, 3), dtype=np.uint8)
                        flag = True
                    else:
                        zeros[:,:,:] = 0
                    zeros[:h, :w, :] = patch_image
                    patch_image = zeros.copy()
                dict[filename_only+"_"+str(idx_x)+"_"+str(idx_y)] = patch_image
    return dict

def save_image(save_folder, file_names, images):
    for filename, image in zip(file_names, images):
        cv2.imwrite(os.path.join(save_folder, filename), image[:,:,::-1])


class TestDataset(Dataset):
    def __init__(self, folder_path):
        data_paths = glob.glob(os.path.join(folder_path, "*.png"))
        dict_ = crop_image(data_paths)
        self.dictionary = dict(dict_)
        self.path_list = sorted(list(self.dictionary.keys()))
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, index):
        ret_path = self.path_list[index]
        ret_img = self.dictionary[ret_path]

        y = int(ret_path.split("_")[-1])
        x = int(ret_path.split("_")[-2])
        image_name = "_".join(ret_path.split("_")[:-2])
        return transforms.ToTensor()(ret_img[:,:,::-1].copy()), y, x, image_name


def main():
    '''Parsing arguments'''
    print("Inference Start..")
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_path", type=str, default='./test_input_img')
    parser.add_argument("--save_path", type=str, default='./best')
    parser.add_argument("--weight_path", type=str, default="./weights_train/model_b3_full_pretrained_142(LB).pt")
    args = parser.parse_args()

    os.makedirs(args.save_path, exist_ok=True)


    '''Prepare the dataset and loader'''
    db_start = time.time()
    test_dataset = TestDataset(args.test_path)
    test_loader = DataLoader(test_dataset, batch_size=21, shuffle=False, pin_memory=True, num_workers=4)
    print("Data Preprocessing...", time.time()-db_start)
    '''Model Setting'''
    setup_start = time.time()
    model = smp.UnetPlusPlus(encoder_name='timm-efficientnet-b3', classes=3)
    model = nn.DataParallel(model, device_ids=[0])
    model.cuda()
    model.load_state_dict(torch.load(args.weight_path))

    print("Model Setup...", time.time() - setup_start)
    condition = {}
    prediction_images = []
    prediction_paths = []
    model.eval()

    patch_making = 0
    output_patch = 0
    inference_only = 0
    sr_only = 0
    start_inf = time.time()
    with torch.no_grad():
        image = torch.zeros((3, 2448, 3264), dtype=torch.float32).cuda()
        for data in test_loader:
            input, y, x, image_name = data
            input = input.cuda()

            with amp.autocast():
                infer_start = time.time()
                model_prediction = model.forward(input)
                prediction = torch.clamp((input + model_prediction), 0., 1.)
                prediction = prediction * 255.0
                infer_end = time.time()
                inference_only += (infer_end - infer_start)


            for idx, (x_, y_, name) in enumerate(zip(x, y, image_name)):

                if name not in condition.keys():
                    condition[name] = np.zeros((max_y, max_x))

                sss = time.time()
                overlap = int(patch_size * (1 - overlap_ratio))
                init_y = int(y_) * overlap
                if int(y_) * overlap < 2448 - patch_size:
                    end_y = int(y_) * overlap + patch_size
                    res_y = patch_size
                else:
                    end_y = 2449
                    res_y = 2448 - (int(y_) * overlap + patch_size)

                init_x = int(x_) * overlap
                if int(x_) * overlap  < 3264 - patch_size:
                    end_x = int(x_) * overlap + patch_size
                    res_x = patch_size
                else:
                    end_x = 3265
                    res_x = 3264 - (int(x_) * overlap + patch_size)

                condition[name][int(y_)][int(x_)] = 1
                eee = time.time()
                patch_making += (eee - sss)
                image[:, init_y: end_y, init_x: end_x] += prediction[idx][:, :res_y, :res_x]


                if np.sum(condition[name]) == max_x * max_y:
                    #to_save_image = to_save_image[:, :2448, :3264].permute(1, 2, 0).cpu().detach().numpy()
                    start_sr = time.time()
                    image[:, 384:, :] /= 2.
                    image[:, :, 384:] /= 2.
                    end_sr = time.time()
                    sr_only += end_sr - start_sr
                    save_name = str(name).replace("_input_", "_") + ".png"
                    s = time.time()

                    prediction_images.append(image.permute(1, 2, 0).byte().detach().cpu().numpy())
                    prediction_paths.append(save_name)
                    end = time.time()
                    print('Inference time per image...', end - start_inf)
                    image[:,:,:] = 0.
                    e = time.time()
                    output_patch += e - s

        save_start = time.time()
        sub_prediction_image = np.array_split(prediction_images, 8)
        sub_prediction_path = np.array_split(prediction_paths, 8)


        process_list = []

        for sub_img, sub_path in zip(sub_prediction_image, sub_prediction_path):
            proc = Process(target=save_image, args=(args.save_path, sub_path, sub_img,))
            process_list.append(proc)
            proc.start()

        for proc in process_list:
            proc.join()
        print("Data Save...", time.time()-save_start)

        os.system('cd '+args.save_path+" "+'&& zip -1 ../submission_upl.zip '+"./*.png")
        print("Inference End")

    ########
    final_end_time = time.time()
    ########
    print("Patch Making:", patch_making)
    print("Type conversion:", output_patch)
    print("Save - Reset:", sr_only, 's')
    print("Model Inference Time:", inference_only, 's')
    print("Inference Elapsed Time:",final_end_time-final_start_time,"s")


if __name__ == "__main__":
    main()
