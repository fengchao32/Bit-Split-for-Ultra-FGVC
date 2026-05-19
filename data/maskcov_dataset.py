import os
import random
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import torch
import torch.utils.data as data
from PIL import Image, ImageStat
from torchvision import transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


_DATASET_INFO = {
    'CUB': ('../Datasets/CUB_200_2011/images', '../Datasets/CUB_200_2011/anno', 200),
    'STCAR': ('../Datasets/st_car/images', '../Datasets/st_car/anno', 196),
    'COTTON': ('../data/COTTON/images', '../data/COTTON/anno', 80),
    'Soybean200': ('../data/soybean200/images', '../data/soybean200/anno', 200),
    'Soybean2000': ('../data/soybean2000/images', '../data/soybean2000/anno', 1938),
    'R1': ('../data/R1/images', '../data/R1/anno', 198),
    'R3': ('../data/R3/images', '../data/R3/anno', 198),
    'R4': ('../data/R4/images', '../data/R4/anno', 198),
    'R5': ('../data/R5/images', '../data/R5/anno', 198),
    'R6': ('../data/R6/images', '../data/R6/anno', 198),
    'soybean_gene': ('../data/soybean_gene/images', '../data/soybean_gene/anno', 1110),
}


def _read_annotation(path):
    rows = []
    with open(path, 'r') as anno_file:
        for line_no, line in enumerate(anno_file, 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                raise ValueError('invalid annotation at {}:{}: {}'.format(path, line_no, line))
            rows.append((parts[0], int(parts[1])))
    return rows


def _random_sample(paths, labels):
    by_label = defaultdict(list)
    for path, label in zip(paths, labels):
        by_label[label].append(path)

    sampled_paths = []
    sampled_labels = []
    for label, label_paths in by_label.items():
        sample_count = max(1, len(label_paths) // 10)
        for idx in random.sample(range(len(label_paths)), sample_count):
            sampled_paths.append(label_paths[idx])
            sampled_labels.append(label)
    return sampled_paths, sampled_labels


@dataclass
class MaskCOVConfig:
    dataset: str
    rawdata_root: str
    anno_root: str
    numcls: int
    swap_num: tuple = (2, 2)
    mask_num: int = 1
    use_cdrm: bool = True
    cls_2: bool = True
    cls_2xmul: bool = False
    crop_resolution: int = 384

    @classmethod
    def from_dataset(
        cls,
        dataset,
        data_root=None,
        num_classes=None,
        swap_num=(2, 2),
        mask_num=1,
        use_cdrm=True,
        cls_2=True,
        cls_2xmul=False,
        crop_resolution=384,
    ):
        if data_root is not None:
            if num_classes is None and dataset in _DATASET_INFO:
                num_classes = _DATASET_INFO[dataset][2]
            if num_classes is None:
                raise ValueError('--num-classes is required when --data-root uses an unknown dataset')
            rawdata_root = os.path.join(data_root, 'images')
            anno_root = os.path.join(data_root, 'anno')
        else:
            if dataset not in _DATASET_INFO:
                raise ValueError('dataset not defined: {}'.format(dataset))
            rawdata_root, anno_root, default_num_classes = _DATASET_INFO[dataset]
            num_classes = num_classes or default_num_classes

        return cls(
            dataset=dataset,
            rawdata_root=rawdata_root,
            anno_root=anno_root,
            numcls=num_classes,
            swap_num=tuple(swap_num),
            mask_num=mask_num,
            use_cdrm=use_cdrm,
            cls_2=cls_2,
            cls_2xmul=cls_2xmul,
            crop_resolution=crop_resolution,
        )

    def annotation(self, split):
        return _read_annotation(os.path.join(self.anno_root, '{}.txt'.format(split)))


class RandomSwap(object):
    def __init__(self, size):
        if isinstance(size, int):
            size = (size, size)
        if len(size) != 2:
            raise ValueError('RandomSwap size must have two dimensions')
        self.size = tuple(size)

    def __call__(self, img):
        return swap(img, self.size)

    def __repr__(self):
        return '{}(size={})'.format(self.__class__.__name__, self.size)


def _resampling_lanczos():
    return getattr(getattr(Image, 'Resampling', Image), 'LANCZOS')


def _crop_image(image, cropnum):
    width, height = image.size
    crop_x = [int((width / cropnum[0]) * i) for i in range(cropnum[0] + 1)]
    crop_y = [int((height / cropnum[1]) * i) for i in range(cropnum[1] + 1)]
    im_list = []
    for j in range(len(crop_y) - 1):
        for i in range(len(crop_x) - 1):
            im_list.append(
                image.crop(
                    (
                        crop_x[i],
                        crop_y[j],
                        min(crop_x[i + 1], width),
                        min(crop_y[j + 1], height),
                    )
                )
            )
    return im_list


def swap(img, crop):
    widthcut, heightcut = img.size
    images = _crop_image(img, crop)

    tmpx = []
    tmpy = []
    count_x = 0
    count_y = 0
    k = 1
    ran = 2
    for i in range(crop[1] * crop[0]):
        tmpx.append(images[i])
        count_x += 1
        if len(tmpx) >= k:
            tmp = tmpx[count_x - ran:count_x]
            random.shuffle(tmp)
            tmpx[count_x - ran:count_x] = tmp
        if count_x == crop[0]:
            tmpy.append(tmpx)
            count_x = 0
            count_y += 1
            tmpx = []
        if len(tmpy) >= k:
            tmp2 = tmpy[count_y - ran:count_y]
            random.shuffle(tmp2)
            tmpy[count_y - ran:count_y] = tmp2

    random_im = []
    for line in tmpy:
        random_im.extend(line)

    width, height = img.size
    iw = int(width / crop[0])
    ih = int(height / crop[1])
    to_image = Image.new('RGB', (iw * crop[0], ih * crop[1]))
    x = 0
    y = 0
    for part in random_im:
        part = part.resize((iw, ih), _resampling_lanczos())
        to_image.paste(part, (x * iw, y * ih))
        x += 1
        if x == crop[0]:
            x = 0
            y += 1

    return to_image.resize((widthcut, heightcut))


def load_data_transformers(resize_reso=440, crop_reso=384, swap_num=(2, 2)):
    normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    return {
        'swap': transforms.Compose([RandomSwap((swap_num[0], swap_num[1]))]),
        'common_aug': transforms.Compose([
            transforms.Resize((resize_reso, resize_reso)),
            transforms.RandomRotation(degrees=15),
            transforms.RandomCrop((crop_reso, crop_reso)),
            transforms.RandomHorizontalFlip(),
        ]),
        'train_totensor': transforms.Compose([
            transforms.Resize((crop_reso, crop_reso)),
            transforms.ToTensor(),
            normalize,
        ]),
        'val_totensor': transforms.Compose([
            transforms.Resize((crop_reso, crop_reso)),
            transforms.ToTensor(),
            normalize,
        ]),
        'test_totensor': transforms.Compose([
            transforms.Resize((crop_reso, crop_reso)),
            transforms.ToTensor(),
            normalize,
        ]),
        'None': None,
    }


class MaskCOVDataset(data.Dataset):
    def __init__(
        self,
        config,
        anno,
        swap_size=(2, 2),
        common_aug=None,
        swap=None,
        totensor=None,
        train=False,
        train_val=False,
        test=False,
    ):
        self.config = config
        self.root_path = config.rawdata_root
        self.numcls = config.numcls
        self.dataset = config.dataset
        self.use_cls_2 = config.cls_2
        self.use_cls_mul = config.cls_2xmul

        self.paths, self.labels = self._normalize_annotation(anno)
        if train_val:
            self.paths, self.labels = _random_sample(self.paths, self.labels)

        self.common_aug = common_aug
        self.swap = swap
        self.totensor = totensor
        self.train = train
        self.swap_size = tuple(swap_size)
        self.test = test

    def _normalize_annotation(self, anno):
        if isinstance(anno, str):
            rows = _read_annotation(anno)
            return [row[0] for row in rows], [row[1] for row in rows]
        if isinstance(anno, dict):
            return list(anno['img_name']), list(anno['label'])
        if hasattr(anno, 'columns') and 'ImageName' in anno.columns and 'label' in anno.columns:
            return anno['ImageName'].tolist(), anno['label'].tolist()
        rows = list(anno)
        return [row[0] for row in rows], [row[1] for row in rows]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, item):
        img_path = os.path.join(self.root_path, self.paths[item])
        img = self.pil_loader(img_path)
        label = self.labels[item] - 1

        if self.test or not self.train:
            img = self.totensor(img) if self.totensor is not None else img
            return img, label, self.paths[item]

        img_unswap = self.common_aug(img) if self.common_aug is not None else img
        width, height = img_unswap.size
        mask_nums = self._sample_mask_indices()
        mask = self._build_mask(width, height, mask_nums)

        image_unswap_list_original = _crop_image(img_unswap, self.swap_size)
        cova_original = [self.cal_covariance(crop_img) for crop_img in image_unswap_list_original]

        img_unswap_mask = Image.fromarray(np.uint8(np.array(img_unswap) * mask))
        image_unswap_list = _crop_image(img_unswap_mask, self.swap_size)
        cova_unswap = [self.cal_covariance(crop_img) for crop_img in image_unswap_list]

        img_swap = self.swap(img_unswap) if self.swap is not None else img_unswap
        image_swap_list = _crop_image(img_swap, self.swap_size)

        unswap_stats = [sum(ImageStat.Stat(im).mean) for im in image_unswap_list_original]
        swap_stats = [sum(ImageStat.Stat(im).mean) for im in image_swap_list]

        cova_swap = []
        for swap_im in swap_stats:
            distance = [abs(swap_im - unswap_im) for unswap_im in unswap_stats]
            index = distance.index(min(distance))
            cova_swap.append(cova_original[index])

        img_swap = Image.fromarray(np.uint8(np.array(img_swap) * mask))
        for mask_index in mask_nums:
            cova_swap[mask_index] = np.zeros((3, 3))

        img_swap = self.totensor(img_swap)
        img_unswap = self.totensor(img_unswap)
        img_unswap_mask = self.totensor(img_unswap_mask)

        if self.use_cls_mul:
            label_swap = label + self.numcls
        elif self.use_cls_2:
            label_swap = -1
        else:
            label_swap = label

        return (
            img_unswap,
            img_unswap_mask,
            img_swap,
            label,
            label,
            label_swap,
            np.array(cova_original).reshape(-1).tolist(),
            np.array(cova_unswap).reshape(-1).tolist(),
            np.array(cova_swap).reshape(-1).tolist(),
            self.paths[item],
        )

    def _sample_mask_indices(self):
        block_count = self.swap_size[0] * self.swap_size[1]
        mask_count = min(self.config.mask_num, block_count)
        return random.sample(range(block_count), mask_count)

    def _build_mask(self, width, height, mask_nums):
        mask = np.ones((height, width, 1), dtype=np.float32)
        block_w = int(width / self.swap_size[0])
        block_h = int(height / self.swap_size[1])
        for mask_num in mask_nums:
            j = mask_num % self.swap_size[0]
            i = mask_num // self.swap_size[0]
            mask[i * block_h:(i + 1) * block_h, j * block_w:(j + 1) * block_w] = 0
        return mask

    def pil_loader(self, imgpath):
        with open(imgpath, 'rb') as img_file:
            with Image.open(img_file) as img:
                return img.convert('RGB')

    def cal_covariance(self, input_img):
        img = np.array(input_img, np.float32) / 255
        h, w, _ = img.shape
        img = img.transpose((2, 0, 1)).reshape((3, -1))
        mean = img.mean(1)
        img = img - mean.reshape(3, 1)
        covariance_matrix = np.matmul(img, np.transpose(img))
        covariance_matrix = covariance_matrix / (h * w - 1)
        return covariance_matrix


def collate_fn4train(batch):
    imgs = []
    labels = []
    labels_swap = []
    law_swap = []
    img_names = []
    for sample in batch:
        imgs.extend([sample[0], sample[1], sample[2]])
        labels.extend([sample[3], sample[3], sample[3]])
        if sample[5] == -1:
            labels_swap.extend([1, 1, 0])
        else:
            labels_swap.extend([sample[3], sample[3], sample[5]])
        law_swap.extend([sample[6], sample[7], sample[8]])
        img_names.append(sample[-1])
    return torch.stack(imgs, 0), labels, labels_swap, law_swap, img_names


def collate_fn4test(batch):
    imgs = []
    labels = []
    img_names = []
    for sample in batch:
        imgs.append(sample[0])
        labels.append(sample[1])
        img_names.append(sample[-1])
    return torch.stack(imgs, 0), labels, img_names


def collate_fn4quant(batch):
    imgs = []
    labels = []
    for sample in batch:
        imgs.append(sample[0])
        labels.append(sample[1])
    return torch.stack(imgs, 0), torch.tensor(labels, dtype=torch.long)


def build_maskcov_quant_loaders(
    config,
    batch_size=16,
    workers=4,
    resize_resolution=440,
    crop_resolution=384,
    train_split='train',
    val_split='val',
):
    transformers = load_data_transformers(resize_resolution, crop_resolution, config.swap_num)

    train_set = MaskCOVDataset(
        config,
        anno=config.annotation(train_split),
        swap_size=config.swap_num,
        common_aug=transformers['None'],
        swap=transformers['None'],
        totensor=transformers['test_totensor'],
        test=True,
    )
    val_set = MaskCOVDataset(
        config,
        anno=config.annotation(val_split),
        swap_size=config.swap_num,
        common_aug=transformers['None'],
        swap=transformers['None'],
        totensor=transformers['test_totensor'],
        test=True,
    )

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
        collate_fn=collate_fn4quant,
        drop_last=False,
    )
    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        collate_fn=collate_fn4quant,
        drop_last=False,
    )
    return train_set, train_loader, val_loader
