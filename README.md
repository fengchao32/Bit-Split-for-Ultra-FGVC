# BitSplit
BitSplit Post-trining Quantization

Code for papers:
* 'Towards Accurate Post-training Network Quantization via Bit-Split and Stitching', ICML 2020

Bit-split is a novel post-training network quantization framework where no finetuning is needed. 



# MaskCOV migration

This fork also registers MaskCOV models and data readers migrated from
`/home/yxh/Downloads/MaskCOV-main/PR_MaskCOV`.

* Model factories: `maskcov_resnet50`, `maskcov_resnet50_quan`,
  `maskcov_resnet101`, `maskcov_resnet101_quan`.
* Data format: `DATA_ROOT/images/...` plus annotation files
  `DATA_ROOT/anno/train.txt`, `val.txt`, `test.txt`; each annotation line is
  `relative/image_name label`, and labels are converted from 1-based to 0-based.
* PTQ entry:

    CUDA_VISIBLE_DEVICES=0 python main_quant_maskcov_twostep.py \
        --dataset soybean_gene \
        --data-root /path/to/soybean_gene \
        -a maskcov_resnet50_quan \
        --pretrained /path/to/best_model.pth \
        --act-bit-width 4 \
        --weight-bit-width 4

* Evaluate a quantized checkpoint:

    CUDA_VISIBLE_DEVICES=0 python main_quant_maskcov_twostep.py \
        --dataset soybean_gene \
        --data-root /path/to/soybean_gene \
        -a maskcov_resnet50_quan \
        --pretrained maskcov_resnet50_quan/soybean_gene_A8W4/state_dict.pth \
        --scales maskcov_resnet50_quan/soybean_gene_A8W4/act_8_scales.npy \
        --evaluate







# Related Papers


    @InProceedings{Wang_2020_ICML,
        author = {Wang, Peisong, Qiang Chen, Xiangyu He, and Cheng, Jian},
        title = {Towards Accurate Post-training Network Quantization via Bit-Split and Stitching},
        booktitle = {Proceedings of the 37nd International Conference on Machine Learning (ICML)},
        month = {July},
        pages = {243--252},
        year = {2020}
    } 
