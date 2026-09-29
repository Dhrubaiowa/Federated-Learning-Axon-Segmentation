# Federated Learning for Axon Segmentation

This repository contains the federated learning implementation used to investigate healthy and damaged axon segmentation across heterogeneous optic nerve histology datasets.

The framework uses an **nnUNet-Lite** segmentation model with **weighted Federated Averaging (FedAvg)** to train a shared global model across multiple clients without directly pooling their local training data. The experiments were designed to evaluate federated learning under heterogeneous and highly imbalanced client data distributions.

This work is part of my research on federated learning for medical image analysis and was developed in conjunction with our study on federated learning for optic nerve axon segmentation.

## Federated Learning Setup

The training data are distributed across six simulated clients originating from three optic nerve histology datasets:

- **Clients 1–2:** Original dataset
- **Clients 3–4:** AxonDeep
- **Clients 5–6:** New Montage

The notebook includes an intentionally imbalanced client configuration in which selected clients contain very limited local training data. This setup was used to investigate the behavior of federated learning under heterogeneous client data distributions.

During each federated round:

1. The current global model is distributed to all clients.
2. Each client performs local training on its own data.
3. Client model parameters are returned to the server.
4. Parameters are aggregated using sample-weighted FedAvg.
5. The updated global model is evaluated on validation data.

Both client-specific and global models are retained for subsequent evaluation.

## Segmentation Task

The model performs three-class semantic segmentation:

| Class | Label |
|---|---:|
| Background | 0 |
| Healthy axon | 1 |
| Damaged axon | 2 |

Healthy and damaged axon annotations are combined into a single semantic label map, with damaged axons taking priority in overlapping regions.

## Model

The federated experiments in this repository use a lightweight U-Net architecture based on nnU-Net design principles.

The main configuration includes:

- Three-class semantic segmentation
- Instance normalization
- LeakyReLU activations
- Encoder-decoder architecture with skip connections
- Cross-entropy and multiclass Dice loss
- AdamW optimization
- Sample-weighted FedAvg aggregation
- Client-specific and global checkpointing
- Early stopping based on global validation performance

## Repository Contents

The main implementation is provided in:

`Federated-Learning-Axon-Segmentation.ipynb`

The notebook includes:

1. Data loading and federated client construction
2. Visualization of client data
3. Data augmentation and client-specific DataLoaders
4. nnUNet-Lite segmentation model
5. Loss functions and evaluation metrics
6. Federated training with weighted FedAvg
7. Local and global model checkpointing
8. Federated training history
9. Cross-dataset evaluation of client and global models
10. Segmentation post-processing
11. Axon counting and instance-level matching
12. Prediction visualization

## Evaluation

The trained client models and global FedAvg model are evaluated across all three test datasets rather than only on the dataset corresponding to each client's local training data.

The evaluation includes:

- Healthy axon Dice
- Damaged axon Dice
- Weighted Dice
- Axon count comparison
- Instance-level matching
- Precision
- Recall
- F1 score

This cross-dataset evaluation was used to study both segmentation performance and generalization across heterogeneous imaging datasets.

## Data

The research datasets are not distributed with this repository.

To use the notebook with another dataset, update the dataset location in the configuration section:

```python
DATA_ROOT = Path("path/to/axon_dataset")
```

The expected dataset organization follows the healthy/damaged annotation structure defined in the data-loading section of the notebook.

## Requirements

The main dependencies include:

- Python
- PyTorch
- NumPy
- OpenCV
- SciPy
- pandas
- Pillow
- Matplotlib
- tqdm

Package dependencies are also provided in `requirements.txt`.

## Related Publication

**Federated learning for optic nerve axon segmentation in mice**  
Durjoy Deb Dhruba, Adam Hedberg-Buenz, Ashelyn Mann, Michael G. Anderson, and Mona Garvin  
*Medical Imaging 2026: Image Processing, Proceedings of SPIE*, 2026.

## Author

**Durjoy Deb Dhruba**  
Ph.D. Candidate, Electrical and Computer Engineering  
University of Iowa
