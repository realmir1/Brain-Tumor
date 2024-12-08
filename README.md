

# BRAIN CANCER CLASSIFICATION USING CONVOLUTIONAL NEURAL NETWORKS (CNN)

This project uses a Convolutional Neural Network (CNN) implemented in TensorFlow/Keras to classify brain cancer types based on medical images. The model identifies three categories of brain cancer: **Brain Glioma**, **Brain Menin**, and **Brain Tumor**.




<p align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/7/7b/Glioma.gif" alt="Resim 1" width="500"/>
  
</p>




## DATASET STRUCTURE

The dataset is expected to follow a directory structure compatible with `ImageDataGenerator`:

```
/Multi Cancer
    /Brain Cancer
        /brain_glioma
            - image1.jpg
            - image2.jpg
            ...
        /brain_menin
            - image1.jpg
            - image2.jpg
            ...
        /brain_tumor
            - image1.jpg
            - image2.jpg
            ...
```



## INSTALLATION

1. Install the required libraries:
   - **TensorFlow**: `pip install tensorflow`
   - **Matplotlib**: `pip install matplotlib`
   - **NumPy**: `pip install numpy`

2. Download the dataset and place it in the correct directory structure as shown above.



## USAGE

1. Place your images in the appropriate folder for each category (`brain_glioma`, `brain_menin`, `brain_tumor`).
2. Run the script to train the model and evaluate performance.



## MODEL ARCHITECTURE

The model consists of the following layers:
1. **Conv2D** layer with 32 filters, 3x3 kernel size, and ReLU activation.
2. **MaxPooling2D** layer with 2x2 pool size.
3. **Conv2D** layer with 64 filters, 3x3 kernel size, and ReLU activation.
4. **MaxPooling2D** layer with 2x2 pool size.
5. **Conv2D** layer with 128 filters, 3x3 kernel size, and ReLU activation.
6. **MaxPooling2D** layer with 2x2 pool size.
7. **Flatten** layer to convert the 2D output into 1D.
8. **Dense** layer with 128 neurons and ReLU activation.
9. **Dense** layer with the number of output neurons equal to the number of classes (3) and softmax activation.


## RESULTS

The script will plot the training and validation accuracy over the epochs and display a sample image with the predicted and true labels.







