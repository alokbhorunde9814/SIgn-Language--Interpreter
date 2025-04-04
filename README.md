# **Sign Language Interpreter**

A Python-based application that utilizes machine learning to detect and interpret American Sign Language (ASL) gestures.

## **Features**

- **ASL Gesture Detection:** Recognizes and interprets ASL gestures in real-time.  
- **Dataset Collection:** Tools to collect and preprocess images for training models.  
- **Model Training:** Scripts to train custom models for ASL detection.  
- **Inference:** Perform real-time inference using trained models.  

## **Prerequisites**

- Python 3.x  
- Required Python packages (listed in `requirements.txt`)  

## **Installation**

1. **Clone the Repository:**  
   ```bash
   git clone https://github.com/alokbhorunde9814/SIgn-Language--Interpreter.git
   cd SIgn-Language--Interpreter
   ```
2. **Install Dependencies:**  
   ```bash
   pip install -r requirements.txt
   ```

## **Usage**

### **1. Collecting Images**
To collect images for training:  
```bash
python collect_imgs.py
```
This script will guide you through capturing images for different ASL gestures.

### **2. Creating Dataset**
After collecting images, preprocess them to create a dataset:  
```bash
python create_dataset.py
```

### **3. Training the Model**
Train the model using the prepared dataset:  
```bash
python train.py
```

### **4. Inference**
To run inference using the trained model:  
```bash
python inference_classifier.py
```
This will start the real-time ASL detection.

## **Files Overview**

- `app.py` - Main application script  
- `collect_imgs.py` - Script to collect images for dataset creation  
- `create_dataset.py` - Preprocesses collected images into a dataset  
- `train.py` - Trains the model using the dataset  
- `train_classifier.py` - Additional training script for classifier models  
- `inference_classifier.py` - Runs inference using the trained classifier model  
- `detect.py` - Contains detection utilities  
- `requirements.txt` - Lists all required Python packages  

## **License**
This project is licensed under the MIT License.
