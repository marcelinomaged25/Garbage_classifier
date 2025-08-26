# Garbage Classifier (Deep Learning Project)

An AI-powered image classification system that sorts garbage into categories using Convolutional Neural Networks (CNNs) and transfer learning with ResNet50.  
This project demonstrates how deep learning can support better waste management and recycling.

---

## Features
- Dataset preprocessing and splitting into train / validation / test sets  
- Data augmentation using `ImageDataGenerator`  
- Transfer learning with ResNet50, fine-tuned for garbage classification  
- Regularization with Batch Normalization, Dropout, and L2  
- Early stopping to prevent overfitting  
- Model evaluation and prediction comparison (True vs Predicted classes)  
- Saves trained model (`.h5`) and weights (locally, excluded from GitHub)  

---

## Project Structure
Garbage_sorter/
│── garabge_sorter.py # Main training & evaluation script
│── Garabge_sorter.ipynb # Jupyter notebook version
│── README.md # Project documentation


---


License
This project is licensed under the MIT License.
Feel free to use, modify, and distribute with attribution.

Author
Youssef Ahmed Mohammed
Email: yousssefa7med@gmail.com
GitHub: Joex-stack
