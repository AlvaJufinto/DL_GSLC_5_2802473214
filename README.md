<!-- @format -->

# Advanced CNNs and Transfer Learning

**Take-Home Assignment, Session 05**

## Student Information

| Field   | Details                 |
| ------- | ----------------------- |
| Name    | Stanislaus Alva Jufinto |
| NIM     | 2802473214              |
| Class   | LA01                    |
| Session | GSLC Session 5          |

## Overview

This project explores transfer learning for image classification with limited training data. Using ResNet-18 and a subset of the Oxford-IIIT Pet dataset, five experiments compare training from scratch, feature extraction, progressive fine-tuning, and the effect of data augmentation.

The goal is to evaluate each strategy using validation accuracy, test Macro F1, training duration, and generalization behavior.

**Repository:** [DL_GSLC_5_2802473214](https://github.com/AlvaJufinto/DL_GSLC_5_2802473214)

## Dataset

The experiment uses 1,981 images from ten pet breeds:

- Abyssinian
- Bengal
- Birman
- Bombay
- British Shorthair
- American Bulldog
- American Pit Bull Terrier
- Basset Hound
- Beagle
- Boxer

The dataset is split using stratified sampling to preserve class proportions.

| Split      | Number of images |
| ---------- | ---------------: |
| Training   |            1,386 |
| Validation |              297 |
| Test       |              298 |
| **Total**  |        **1,981** |

## Experimental Setup

All experiments use ResNet-18 with ImageNet-pretrained weights where applicable.

| Setting              | Configuration                    |
| -------------------- | -------------------------------- |
| Architecture         | ResNet-18                        |
| Pretrained weights   | `ResNet18_Weights.IMAGENET1K_V1` |
| Input size           | 224 × 224                        |
| Normalization        | ImageNet normalization           |
| Batch size           | 32                               |
| Optimizer            | AdamW                            |
| Loss function        | `CrossEntropyLoss`               |
| Random seed          | 42                               |
| Checkpoint selection | Lowest validation loss           |
| Early stopping       | Patience of 4 epochs             |

The test set is used for final evaluation after selecting the best checkpoint using validation loss.

## Experiments

| Experiment | Strategy                                   | Trainable parameters | Learning rate              |
| ---------- | ------------------------------------------ | -------------------: | -------------------------- |
| B1         | Train from scratch, without augmentation   |           11,181,642 | 1e-3                       |
| B2         | Feature extraction, with augmentation      |                5,130 | 1e-3                       |
| B3         | Progressive fine-tuning, with augmentation |            8,398,858 | `layer4`: 1e-4, `fc`: 1e-3 |
| B4a        | Feature extraction, without augmentation   |                5,130 | 1e-3                       |
| B4b        | Feature extraction, with augmentation      |                5,130 | 1e-3                       |

B1 trains the entire network from random initialization. B2 freezes the pretrained backbone and trains only the classifier head. B3 additionally fine-tunes `layer4`. B4a and B4b compare feature extraction without and with data augmentation under the same general training strategy.

## Results

| Experiment | Best epoch / Epochs run | Training time | Validation accuracy | Test Macro F1 |
| ---------- | ----------------------: | ------------: | ------------------: | ------------: |
| B1         |                  8 / 12 |       474.2 s |              39.06% |        36.93% |
| B2         |                 12 / 16 |       444.7 s |              93.27% |        90.59% |
| B3         |                   4 / 8 |       194.2 s |          **95.29%** |        92.25% |
| B4a        |                 12 / 16 |       216.2 s |              93.94% |    **92.65%** |
| B4b        |                 12 / 16 |       273.1 s |              93.27% |        90.59% |

The results show that transfer learning substantially outperforms training from scratch in this limited-data setting. B3 achieves the highest validation accuracy, while B4a achieves the highest test Macro F1.

### Training and Validation Loss

The following figure compares the training and validation loss of B1, B2, and B3.

![Training and validation loss curves for B1, B2, and B3](artifacts/loss_curves_b1_b2_b3.png)

_Figure 1. Training and validation loss curves for the scratch baseline, feature extraction, and progressive fine-tuning experiments._

### Generalization Gap

The next figure compares the generalization gap for B4a and B4b. The gap is calculated as the difference between training accuracy and validation accuracy.

![Generalization gap comparison for B4](artifacts/b4_generalization_gap.png)

_Figure 2. Generalization gap across epochs for feature extraction with and without data augmentation._

## Key Findings

### 1. Transfer learning versus training from scratch

B1 achieves a test Macro F1 of only 36.93%, compared with 90.59% for B2. The pretrained backbone provides useful visual representations that can be adapted to the target classes without learning all features from random initialization.

### 2. Feature extraction versus progressive fine-tuning

B3 increases test Macro F1 from 90.59% to 92.25% compared with B2, an improvement of 1.66 percentage points. It also achieves the highest validation accuracy at 95.29%.

However, B3 trains approximately 8.4 million parameters, compared with only 5,130 parameters in B2. Fine-tuning can improve task-specific representations, but it requires updating substantially more parameters.

### 3. Effect of data augmentation

B4a achieves a test Macro F1 of 92.65%, while B4b achieves 90.59%. In this particular configuration, augmentation does not improve test performance.

This result does not establish that augmentation is generally harmful. The experiments use one random seed and one augmentation configuration, so further testing would be needed to determine whether the difference is consistent.

### 4. Training efficiency

B3 has the shortest total training duration at 194.2 seconds, but it also runs for only eight epochs. Total duration alone is therefore insufficient to conclude that progressive fine-tuning is faster per epoch.

Feature extraction remains attractive when training simplicity and the number of trainable parameters are important considerations.

## Conclusion

For this dataset and experimental protocol, **B4a, feature extraction without data augmentation, achieves the highest test Macro F1 at 92.65%**. It trains only the classifier head while keeping the pretrained backbone frozen.

B3 is another strong option, achieving the highest validation accuracy of 95.29% and a test Macro F1 of 92.25%. The choice between these configurations depends on whether test Macro F1, validation performance, or training complexity is prioritized.

These findings are limited to the ten selected classes and the configurations tested. They should not be interpreted as proof that one strategy is universally best for pet image classification.

## Repository Structure

```text
DL_GSLC_5_2802473214/
├── artifacts/
│   ├── checkpoints/
│   ├── B1_scratch_epoch_metrics.csv
│   ├── B2_feature_extraction_epoch_metrics.csv
│   ├── B3_progressive_finetuning_epoch_metrics.csv
│   ├── B4_no_augmentation_epoch_metrics.csv
│   ├── B4_with_augmentation_epoch_metrics.csv
│   ├── experiment_summary.csv
│   ├── experiment_summary.json
│   ├── split_manifest.csv
│   ├── loss_curves_b1_b2_b3.png
│   └── b4_generalization_gap.png
├── BAGIAN_A.md
├── BAGIAN_C.md
├── experiment.ipynb
├── questions.md
└── README.md
```

## Project Files

- [`experiment.ipynb`](experiment.ipynb): Notebook containing the experiments and evaluation workflow.
- [`BAGIAN_A.md`](BAGIAN_A.md): Assignment analysis and conceptual discussion.
- [`BAGIAN_C.md`](BAGIAN_C.md): Experimental results, comparative analysis, and engineering recommendations.
- [`questions.md`](questions.md): Assignment questions.
- [`artifacts/`](artifacts/): Experiment metrics, summaries, dataset split manifest, checkpoints, and figures.

## Reproducibility and Limitations

The repository includes experiment summaries, per-epoch metrics, and a dataset split manifest to support result verification. Training duration depends on the hardware and execution environment.

The experiments have several limitations:

- Only ten of the 37 Oxford-IIIT Pet classes are included.
- Results are based on a single random seed.
- Hyperparameter and augmentation searches are limited.
- Training duration is hardware-dependent.
- Aggregate metrics alone cannot establish the visual causes of individual misclassifications.

Failure analysis should be completed using actual test predictions and images before drawing conclusions about specific model errors.

## AI Usage Disclosure

AI tools were used to assist with code structure, debugging, and report organization. Experimental metrics were checked against the saved results in the repository's `artifacts/` directory.

## References

- [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/)
- [PyTorch ResNet-18 Documentation](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html)
- [PyTorch Transfer Learning Tutorial](https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)
