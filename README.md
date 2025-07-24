# Brain Segmentation Project

This repository contains a comprehensive implementation of brain image segmentation using various deep learning architectures.

## Author

Christian JefreyFonseca Rodriguez

## Project Overview

This project focuses on segmenting brain images using state-of-the-art deep learning models. The main implementation is contained in `book_v2.ipynb`, with specialized versions for different architectures.

## Models Implemented

The project implements and evaluates several deep learning architectures for brain segmentation:
- U-Net (ResNet50 backbone)
- Attention U-Net
- DeepLabv3+
- PSPNet
- SegFormer

## Dependencies

The following dependencies are required to run this project:
```
numpy
scipy
tqdm
scikit-learn
opencv-python
matplotlib
grad-cam
segmentation-models-pytorch
nibabel
```

Install PyTorch with CUDA support:
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

Install the other dependencies using:
```
pip install -r requirements.txt
```

## Data

The project uses the BraTS 2021 Task 1 dataset. You need to download the data from Kaggle:

[BraTS 2021 Task 1 Dataset](https://www.kaggle.com/datasets/dschettler8845/brats-2021-task1/)

After downloading, extract the dataset to the `data/` directory in the project root.

## Project Structure

- `book_v2.ipynb`: Main notebook containing the complete implementation
- Model-specific notebooks:
  - `book_v2 - Unet.ipynb`: Implementation using U-Net architecture
  - `book_v2 - Attention.ipynb`: Implementation using Attention U-Net
  - `book_v2 - DeepLabv3.ipynb`: Implementation using DeepLabv3+
  - `book_v2 - PSPNet.ipynb`: Implementation using PSPNet
  - `book_v2 - SegFormer.ipynb`: Implementation using SegFormer
- `data/`: Directory containing the brain imaging datasets
- Pre-trained models:
  - `best_model.pth`: Default best model
  - `best_model_attention_u-net.pth`
  - `best_model_deeplabv3plus.pth`
  - `best_model_pspnet.pth`
  - `best_model_segformer.pth`
  - `best_model_u-net_(resnet50).pth`
- `results_history_*.pkl`: Pickle files containing training history for each model

## Usage

1. Install the required dependencies
2. Open the main notebook `book_v2.ipynb` or any of the model-specific notebooks
3. Follow the instructions within the notebook to train models or perform inference

## Results

The project includes trained models and their performance history, saved as pickle files. These can be loaded to visualize the training progress and final performance metrics.

## Analysis

For additional analysis of the results, refer to the `analysis.ipynb` notebook.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
