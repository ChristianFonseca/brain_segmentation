# Proyecto de Segmentación Cerebral / Brain Segmentation Project

*[English version below](#english-version)*

## Español

### Descripción del Proyecto

Este repositorio contiene una implementación completa de segmentación de imágenes cerebrales utilizando diversas arquitecturas de aprendizaje profundo.

### Autor

Christian Jefrey Fonseca Rodriguez

### Visión General del Proyecto

Este proyecto se centra en la segmentación de imágenes cerebrales utilizando modelos de aprendizaje profundo de última generación. La implementación principal se encuentra en `book_v2.ipynb`, con versiones especializadas para diferentes arquitecturas.

### Modelos Implementados

El proyecto implementa y evalúa varias arquitecturas de aprendizaje profundo para la segmentación cerebral:
- U-Net (con backbone ResNet50)
- Attention U-Net
- DeepLabv3+
- PSPNet
- SegFormer

### Dependencias

Las siguientes dependencias son necesarias para ejecutar este proyecto:
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

Instalar PyTorch con soporte CUDA:
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

Instalar las demás dependencias usando:
```
pip install -r requirements.txt
```

### Datos

El proyecto utiliza el conjunto de datos BraTS 2021 Task 1. Necesitas descargar los datos desde Kaggle:

[Conjunto de datos BraTS 2021 Task 1](https://www.kaggle.com/datasets/dschettler8845/brats-2021-task1/)

Después de descargar, extrae el conjunto de datos en el directorio `data/` en la raíz del proyecto.

### Estructura del Proyecto

- `book_v2.ipynb`: Notebook principal que contiene la implementación completa
- Notebooks específicos por modelo:
  - `book_v2 - Unet.ipynb`: Implementación usando la arquitectura U-Net
  - `book_v2 - Attention.ipynb`: Implementación usando Attention U-Net
  - `book_v2 - DeepLabv3.ipynb`: Implementación usando DeepLabv3+
  - `book_v2 - PSPNet.ipynb`: Implementación usando PSPNet
  - `book_v2 - SegFormer.ipynb`: Implementación usando SegFormer
- `data/`: Directorio que contiene los conjuntos de datos de imágenes cerebrales
- Modelos pre-entrenados:
  - `best_model.pth`: Mejor modelo predeterminado
  - `best_model_attention_u-net.pth`
  - `best_model_deeplabv3plus.pth`
  - `best_model_pspnet.pth`
  - `best_model_segformer.pth`
  - `best_model_u-net_(resnet50).pth`
- `results_history_*.pkl`: Archivos pickle que contienen el historial de entrenamiento para cada modelo

### Uso

1. Instala las dependencias requeridas
2. Abre el notebook principal `book_v2.ipynb` o cualquiera de los notebooks específicos por modelo
3. Sigue las instrucciones dentro del notebook para entrenar modelos o realizar inferencia

### Resultados

El proyecto incluye modelos entrenados y su historial de rendimiento, guardados como archivos pickle. Estos pueden cargarse para visualizar el progreso del entrenamiento y las métricas de rendimiento final.

### Análisis

Para análisis adicionales de los resultados, consulta el notebook `analysis.ipynb`.

### Licencia

Este proyecto está licenciado bajo la Licencia MIT - consulta el archivo [LICENSE](LICENSE) para más detalles.

---

<a name="english-version"></a>
## English

### Project Overview

This repository contains a comprehensive implementation of brain image segmentation using various deep learning architectures.

### Author

Christian Jefrey Fonseca Rodriguez

### Project Description

This project focuses on segmenting brain images using state-of-the-art deep learning models. The main implementation is contained in `book_v2.ipynb`, with specialized versions for different architectures.

### Models Implemented

The project implements and evaluates several deep learning architectures for brain segmentation:
- U-Net (ResNet50 backbone)
- Attention U-Net
- DeepLabv3+
- PSPNet
- SegFormer

### Dependencies

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

### Data

The project uses the BraTS 2021 Task 1 dataset. You need to download the data from Kaggle:

[BraTS 2021 Task 1 Dataset](https://www.kaggle.com/datasets/dschettler8845/brats-2021-task1/)

After downloading, extract the dataset to the `data/` directory in the project root.

### Project Structure

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

### Usage

1. Install the required dependencies
2. Open the main notebook `book_v2.ipynb` or any of the model-specific notebooks
3. Follow the instructions within the notebook to train models or perform inference

### Results

The project includes trained models and their performance history, saved as pickle files. These can be loaded to visualize the training progress and final performance metrics.

### Analysis

For additional analysis of the results, refer to the `analysis.ipynb` notebook.

### License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
