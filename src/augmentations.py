import cv2
import numpy as np
import torch
import albumentations as A
from albumentations.core.transforms_interface import DualTransform
import os

# 출처: https://www.kaggle.com/shivyshiv/efficientnet-gridmask-training-pytorch
class GridMask_(DualTransform):
    """GridMask augmentation for image classification and object detection.

    Args:
        num_grid (int): number of grid in a row or column.
        fill_value (int, float, lisf of int, list of float): value for dropped pixels.
        rotate ((int, int) or int): range from which a random angle is picked. If rotate is a single int
            an angle is picked from (-rotate, rotate). Default: (-90, 90)
        mode (int):
            0 - cropout a quarter of the square of each grid (left top)
            1 - reserve a quarter of the square of each grid (left top)
            2 - cropout 2 quarter of the square of each grid (left top & right bottom)

    Targets:
        image, mask

    Image types:
        uint8, float32

    Reference:
    |  https://arxiv.org/abs/2001.04086
    |  https://github.com/akuxcw/GridMask
    """

    def __init__(self, num_grid=3, fill_value=0, rotate=0, mode=0, always_apply=False, p=0.5):
        super(GridMask_, self).__init__(always_apply, p)
        if isinstance(num_grid, int):
            num_grid = (num_grid, num_grid)
        if isinstance(rotate, int):
            rotate = (-rotate, rotate)
        self.num_grid = num_grid
        self.fill_value = fill_value
        self.rotate = rotate
        self.mode = mode
        self.masks = None
        self.rand_h_max = []
        self.rand_w_max = []

    def init_masks(self, height, width):
        if self.masks is None:
            self.masks = []
            n_masks = self.num_grid[1] - self.num_grid[0] + 1
            for n, n_g in enumerate(range(self.num_grid[0], self.num_grid[1] + 1, 1)):
                grid_h = height / n_g
                grid_w = width / n_g
                this_mask = np.ones((int((n_g + 1) * grid_h), int((n_g + 1) * grid_w))).astype(np.uint8)
                for i in range(n_g + 1):
                    for j in range(n_g + 1):
                        this_mask[
                        int(i * grid_h): int(i * grid_h + grid_h / 2),
                        int(j * grid_w): int(j * grid_w + grid_w / 2)
                        ] = self.fill_value
                        if self.mode == 2:
                            this_mask[
                            int(i * grid_h + grid_h / 2): int(i * grid_h + grid_h),
                            int(j * grid_w + grid_w / 2): int(j * grid_w + grid_w)
                            ] = self.fill_value

                if self.mode == 1:
                    this_mask = 1 - this_mask

                self.masks.append(this_mask)
                self.rand_h_max.append(grid_h)
                self.rand_w_max.append(grid_w)

    def apply(self, image, mask, rand_h, rand_w, angle, **params):
        h, w = image.shape[:2]
        hm, wm = mask.shape[:2]
        # angle = np.random.uniform(-angle, angle)
        # matrix = cv2.getRotationMatrix2D((wm/2, hm/2), angle, 1)
        # mask = cv2.warpAffine(mask, matrix, (int(wm*1.5), int(hm*1.5)))
        #mask = cv2.rotate(mask, angle) if self.rotate[1] > 0 else mask
        mask = mask[:, :, np.newaxis] if image.ndim == 3 else mask
        image *= mask[rand_h:rand_h + h, rand_w:rand_w + w].astype(image.dtype)
        return image

    def get_params_dependent_on_targets(self, params):
        img = params['image']
        height, width = img.shape[:2]
        self.init_masks(height, width)

        mid = np.random.randint(len(self.masks))
        mask = self.masks[mid]
        rand_h = np.random.randint(self.rand_h_max[mid])
        rand_w = np.random.randint(self.rand_w_max[mid])
        angle = np.random.randint(self.rotate[0], self.rotate[1]) if self.rotate[1] > 0 else 0

        return {'mask': mask, 'rand_h': rand_h, 'rand_w': rand_w, 'angle': angle}

    @property
    def targets_as_params(self):
        return ['image']

    def get_transform_init_args_names(self):
        return ('num_grid', 'fill_value', 'rotate', 'mode')


def shear_transfrom(img1, img2=None, strength=0.5, is_vertical=True, p=0.3):
    if np.random.rand() < p:
        h, w, c = img1.shape
        src = np.float32([[w / 3, h / 3], [2 * w / 3, h / 3], [w / 3, 2 * h / 3]])

        degree = np.random.uniform(-strength, strength)
        if is_vertical:
            dst = np.float32([[w / 3, h / 3 + (h / 3) * degree], [2 * w / 3, h / 3], [w / 3, 2 * h / 3 + (h / 3) * degree]])
        else:
            dst = np.float32(
                [[(w / 3) + (w / 3) * degree, h / 3], [(2 * w / 3) + degree * (w / 3), h / 3], [(w / 3), 2 * h / 3]])

        M = cv2.getAffineTransform(src, dst)
        img1 = cv2.warpAffine(img1, M, (w, h))
        if img2 is not None:
            img2 = cv2.warpAffine(img2, M, (w, h))
            return img1, img2
        else:
            return img1
    else:
        return img1, img2 if img2 is not None else img1


class ShearTransform:
    def __init__(self, strength = 0.5, p=0.5):
        self.p = p
        self.strength = strength
    def __call__(self, img1, img2=None):
        return shear_transfrom(img1, img2, self.strength, p=self.p)


class HorizontalFlip:
    def __init__(self, p=0.5):
        self.hflip = A.Compose(
            [A.HorizontalFlip(p=p)],
            additional_targets={"target_image": "image"})

    def __call__(self, img1, img2=None):
        if img2 is None:
            img1 = self.hflip(image=img1)['image']
            return img1
        else:
            transformed = self.hflip(image=img1, target_image=img2)
            img1 = transformed['image']
            img2 = transformed['target_image']
            return img1, img2


class VerticalFlip:
    def __init__(self, p=0.5):
        self.vflip = A.Compose(
            [A.VerticalFlip(p=p)],
            additional_targets={"target_image": "image"})

    def __call__(self, img1, img2=None):
        if img2 is None:
            img1 = self.vflip(image=img1)['image']
            return img1
        else:
            transformed = self.vflip(image=img1, target_image=img2)
            img1 = transformed['image']
            img2 = transformed['target_image']
            return img1, img2


class CutOut:
    def __init__(self, p=0.5):
        self.cutout = A.Compose(
            [A.Cutout(num_holes=8, max_h_size=86, max_w_size=86, fill_value=0, always_apply=False, p=p)],
            additional_targets={"target_image": "image"})

    def __call__(self, img1, img2=None):
        if img2 is None:
            img1 = self.cutout(image=img1)['image']
            return img1
        else:
            transformed = self.cutout(image=img1, target_image=img2)
            img1 = transformed['image']
            img2 = transformed['target_image']
            return img1, img2


class GridMask:
    def __init__(self, p=0.5):
        self.gm = A.Compose([
        GridMask_(num_grid=4, mode=0, rotate=30, p=p, fill_value=0.)
    ], additional_targets={"target_image": "image"})

    def __call__(self, img1, img2=None):
        if img2 is None:
            img1 = self.gm(image=img1)['image']
            return img1
        else:
            transformed = self.gm(image=img1, target_image=img2)
            img1 = transformed['image']
            img2 = transformed['target_image']
            return img1, img2


if __name__ == "__main__":
    sample_img = cv2.imread('./sample_img.png')
    co = CutOut(p=1.)
    hf = HorizontalFlip(p=1.)
    vf = VerticalFlip(p=1.)
    gm = GridMask(p=1.)
    sh = ShearTransform(p=1.)



    sample_img_hflip = hf(sample_img.copy())
    sample_img_vflip = vf(sample_img.copy())
    sample_img_shear = sh(sample_img.copy())
    sample_img_gridmask = gm(sample_img.copy())
    sample_img_cutout = co(sample_img.copy())

    save_path = './augmentation_save'
    os.makedirs(save_path, exist_ok=True)

    cv2.imwrite(os.path.join(save_path, "sample_hflip.png"), sample_img_hflip)
    cv2.imwrite(os.path.join(save_path, "sample_vflip.png"), sample_img_vflip)
    cv2.imwrite(os.path.join(save_path, "sample_shear.png"), sample_img_shear)
    cv2.imwrite(os.path.join(save_path, "sample_gridmask.png"), sample_img_gridmask)
    cv2.imwrite(os.path.join(save_path, "sample_cutout.png"), sample_img_cutout)