# Satellite Image Classification Project

## Project Overview
This project implements a Convolutional Neural Network (CNN) for classifying satellite imagery. It processes multi-channel satellite images and predicts corresponding labels, which could represent various land use categories or other features of interest.

## Key Features
- Data ingestion and preprocessing of multi-channel satellite images
- Custom CNN architecture designed for satellite imagery
- Training pipeline with data augmentation and class balancing
- Performance visualization and evaluation metrics

## Dependencies
- TensorFlow
- NumPy
- scikit-learn
- scikit-image
- Matplotlib

## Project Structure
```
Project_Name/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   ├── exploration.ipynb
│   └── model_training.ipynb
├── src/
│   ├── data_preprocessing.py
│   └── model.py
├── tests/
│   ├── test_data_preprocessing.py
│   └── test_model.py
├── README.md
├── requirements.txt
└── setup.py
```

### Directory Descriptions:
- `data/`: Contains raw and processed datasets
- `notebooks/`: Jupyter notebooks for data exploration and model training
- `src/`: Source code for data preprocessing and model definition
- `tests/`: Unit tests for the project components
- `README.md`: Project documentation (this file)
- `requirements.txt`: List of project dependencies
- `setup.py`: Script for setting up the project

## Components
1. **Data Ingestion**: 
   - Loads and preprocesses satellite images from the `data/raw/` directory
   - Handles multi-channel images (12 channels) and corresponding labels
   - Processed data is stored in `data/processed/`

2. **Data Preparation**:
   - Splits data into training, validation, and test sets
   - Implements a custom data generator for efficient batch processing

3. **Model Architecture**:
   - Defines a CNN model using TensorFlow/Keras in `src/model.py`
   - Includes convolutional layers, batch normalization, and dropout for regularization

4. **Model Training**:
   - Training process is documented in `notebooks/model_training.ipynb`
   - Compiles the model with appropriate loss function and optimizer
   - Implements early stopping and learning rate reduction strategies
   - Uses class weights to handle potential class imbalance

5. **Performance Visualization**:
   - Plots training history (accuracy and loss)
   - Generates precision-recall and ROC curves
   - Visualizes sample predictions against ground truth

## Usage
1. Ensure all dependencies are installed: `pip install -r requirements.txt`
2. Run `setup.py` to set up the project environment
3. Place your raw data in the `data/raw/` directory
4. Run data preprocessing scripts from `src/data_preprocessing.py`
5. Use notebooks in the `notebooks/` directory for exploration and model training
6. Execute tests using `python -m unittest discover tests`

## Data Format
The script expects data in the following format in `data/raw/`:
- A main directory containing subdirectories for each timestamp
- Each timestamp directory should contain:
  - 12 channel images named `channel_0.png` to `channel_11.png`
  - A label image named `label.png`

## Model Output
The model predicts a 32x32 label map for each input stack of 12 satellite image channels.

## Performance Metrics
The script calculates and reports:
- Test Accuracy
- Test Precision
- Test Recall
- Best Validation Accuracy
- Best Validation Loss

## Visualizations
The script generates several plots to help understand model performance:
- Training and validation accuracy/loss curves
- Precision-Recall curve
- ROC curve
- Sample predictions compared to ground truth

## Note
This project is designed for research and educational purposes. Ensure you have the necessary rights to use any satellite imagery data for training and testing.
