# README: Pneumonia Detection Using Chest X-rays with TensorFlow

This project utilizes TensorFlow to build, train, and evaluate a convolutional neural network (CNN) for detecting pneumonia from chest X-ray images. Below is a detailed explanation of the project structure, setup, and usage.

---

## Project Structure

- **Dataset Directories**:
  - `train`: Contains training images.
  - `test`: Contains testing images.
  - `val`: Contains validation images.

- **Python Script**:
  The script is designed to load and preprocess the dataset, define and train a CNN model, and evaluate its performance on unseen test data.

---

## Prerequisites

Before running the project, ensure the following:

1. **Python**: Install Python (3.8 or later).
2. **TensorFlow**: Install TensorFlow.
3. **Dataset**: Ensure the dataset is available in the appropriate directories as defined in the script.

### Installing Dependencies
Run the following command to install necessary Python libraries:
```bash
pip install tensorflow
```

---

## Dataset Preparation

The dataset should have the following structure:
```
chest_xray_pneumonia/
  |-- train/
  |-- test/
  |-- val/
```
- Each folder should contain subfolders for the two classes (`NORMAL` and `PNEUMONIA`).

---

## Running the Code

1. Clone or download the repository.
2. Update the dataset directory paths in the script:
   ```python
   train_dir = "C:/path_to_train"
   test_dir = "C:/path_to_test"
   val_dir = "C:/path_to_val"
   ```
3. Execute the script using:
   ```bash
   python app.py
   ```

---

## Workflow

1. **Dataset Loading and Preprocessing**
   - Images are loaded using TensorFlow's `image_dataset_from_directory`.
   - Rescaling is applied to normalize pixel values to the range [0,1].

2. **Model Architecture**
   - A CNN is defined with the following layers:
     - Rescaling Layer
     - 3 Convolutional Blocks (Conv2D + MaxPooling2D)
     - Flatten Layer
     - Dense Layers (Fully connected)
   - Output Layer uses a sigmoid activation function for binary classification.

3. **Training**
   - The model is compiled with the Adam optimizer and binary cross-entropy loss function.
   - The model is trained for 10 epochs using the training and validation datasets.

4. **Evaluation**
   - The model's accuracy is evaluated on the test dataset.

5. **Saving the Model**
   - The trained model is saved as `chest_xray_model.h5`.

---

## Results

- The model's performance on the test dataset is printed in terms of accuracy.
- Example output:
  ```
  Test accuracy: 0.95
  ```

---

## Troubleshooting

- **Dataset Path Issues**:
  Ensure the paths to the dataset directories are correct and accessible.

- **Memory Errors**:
  Reduce the batch size or use a smaller image size if memory issues occur.

---

## Acknowledgments
- Dataset: Chest X-Ray Images (Pneumonia) from [Kaggle](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia).
- Framework: TensorFlow.

