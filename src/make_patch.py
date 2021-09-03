import cv2
import numpy as np
import glob
import os
import shutil

import argparse

def crop_image(origin_filename, save_folder,patch_size=768, overlap=0.5):
    origin_image = cv2.imread(origin_filename)
    filename_only = origin_filename.split("/")[-1].split(".")[0]
    x_list = [i for i in range(0, len(origin_image[0]), int(patch_size * (1 - overlap)))]
    y_list = [i for i in range(0, len(origin_image), int(patch_size * (1 - overlap)))]
    os.makedirs(save_folder, exist_ok=True)
    print(origin_filename)
    for idx_x, x in enumerate(x_list):
        for idx_y, y in enumerate(y_list):
            if os.path.exists(os.path.join(save_folder, filename_only + "_" + str(idx_x) + "_" + str(idx_y) + ".png")):
                continue
            patch_image = origin_image[y:y+patch_size, x:x+patch_size, :]
            if patch_image.shape != (patch_size, patch_size, 3):
                h, w, c  = patch_image.shape
                zeros = np.zeros((patch_size, patch_size, 3), dtype=np.uint8)
                zeros[:h, :w, :] = patch_image
                patch_image = zeros.copy()
            cv2.imwrite(os.path.join(save_folder, filename_only+"_"+str(idx_x)+"_"+str(idx_y)+".png"), patch_image)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_input_img", type=str, default='/home/leehanbeen/data/train_input_img')
    parser.add_argument("--train_label_img", type=str, default='/home/leehanbeen/data/train_label_img')
    parser.add_argument('--to_save_folder', type=str, default='/home/leehanbeen/data/to_save_folder')

    args = parser.parse_args()

    os.makedirs(args.to_save_folder, exist_ok=True)
    os.makedirs(os.path.join(args.to_save_folder, "train_input_image_3200_only"), exist_ok=True)
    os.makedirs(os.path.join(args.to_save_folder, "train_label_image_3200_only"), exist_ok=True)
    os.makedirs(os.path.join(args.to_save_folder, "valid_input_image_3200_ol2"), exist_ok=True)
    os.makedirs(os.path.join(args.to_save_folder, "valid_label_image_3200_ol2"), exist_ok=True)
    os.makedirs(os.path.join(args.to_save_folder, "valid_input_image_3200_only"), exist_ok=True)
    os.makedirs(os.path.join(args.to_save_folder, "valid_label_image_3200_only"), exist_ok=True)

    train_images = sorted(glob.glob(os.path.join(args.train_input_img, "*.png")))
    train_labels = sorted(glob.glob(os.path.join(args.train_label_img, "*.png")))

    valid_input_idx = [10001, 10008, 10015, 10022, 10029, 10036, 10044, 10051, 10057, 10067, 10074, 10084, 10094, 10100, 10107, 10115,
                       10124, 10132, 10138, 10146, 10153, 10160, 10168, 10174, 10180, 10190, 10201, 10208, 10215, 10223, 10230, 10237,
                       10245, 10253, 10258, 10264]

    for image, label in zip(train_images, train_labels):
        print(image)

        if int(image.split("/")[-1].split(".")[0].split("_")[-1]) >= 10272:
            continue
        if int(image.split("/")[-1].split(".")[0].split("_")[-1]) in valid_input_idx:
            crop_image(image, os.path.join(args.to_save_folder, "valid_input_image_3200_ol2"), overlap=0.5)
            crop_image(label, os.path.join(args.to_save_folder, "valid_label_image_3200_ol2"), overlap=0.5)
            crop_image(image, os.path.join(args.to_save_folder, "valid_input_image_3200_only"), overlap=0.)
            crop_image(label, os.path.join(args.to_save_folder, "valid_label_image_3200_only"), overlap=0.)
        else:
            crop_image(image, os.path.join(args.to_save_folder, "train_input_image_3200_only"), overlap=0.5)
            crop_image (label, os.path.join (args.to_save_folder, "train_label_image_3200_only"), overlap=0.5)



