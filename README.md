# # 🌱 AI-Powered-Plant-Disease-Detector

## Dataset  
The dataset is hosted on Google Drive due to size limitations. Download it here:  
[Download Dataset](https://drive.google.com/drive/folders/1Pouiw4pED2L5n1OzBFou23J2fap4eiF6?usp=sharing)

Trained Model: https://drive.google.com/file/d/1-1JOnLrccuDD3N9P7AtNM2XVhY_1K3_D/view?usp=drive_link
https://drive.google.com/file/d/1-1BCtKalza5IMp8UoYwkqjDCKoR2XxfC/view?usp=drive_link

necessary 

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12%2B-orange)
![Gradio](https://img.shields.io/badge/UI-Gradio-blueviolet)

A Convolutional Neural Network (CNN) model trained to classify plant diseases from leaf images. Built with TensorFlow/Keras and deployed via Gradio.

---

## 🚀 Features
- **4-Class Classification**: Detects `Blight`, `Common Rust`, `Gray Leaf Spot`, and `Healthy` leaves.
- **Web Interface**: Gradio UI for real-time predictions.
- **Model Accuracy**: ~94% validation accuracy (customizable with your metrics).

---

## 🛠️ Setup (Google Colab)
1. **Mount Google Drive**:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
Install Dependencies:

python

!pip install tensorflow keras numpy matplotlib gradio

🧠 Model Architecture
python

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense

model = Sequential([
    Input(shape=(224, 224, 3)),
    Conv2D(32, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(4, activation='softmax')  # 4 classes
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
📂 Dataset
Source: PlantVillage Dataset (Kaggle)

Structure:


/content/drive/MyDrive/plant_dataset/train/
├── Blight/
├── Common_Rust/
├── Gray_Leaf_Spot/
└── Healthy/
Download: Google Drive Backup (if using your dataset)

🖥️ Usage
Training
python

history = model.fit(
    train_data,
    epochs=10,
    validation_data=val_data
)
model.save('/content/drive/MyDrive/plant_dataset/model.h5')
Gradio Deployment
python

import gradio as gr

def predict_image(img):
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    prediction = model.predict(np.expand_dims(img_array, axis=0))
    return list(train_data.class_indices.keys())[np.argmax(prediction)]

gr.Interface(fn=predict_image, inputs="image", outputs="label").launch(share=True)
📊 Results
Metric	Training	Validation
Accuracy	98.5%	94.2%
Loss	0.05	0.19
Training Plot Replace with your plot

🌐 Deployment
Hugging Face Spaces:
Hugging Face



python predict.py --image test_image.jpg

🙏 Acknowledgments

Guided Implementation: Built with assistance from an AI mentor.

Tools: TensorFlow, Google Colab, Gradio.

Dataset: PlantVillage (Kaggle).

📜 License
MIT License - See LICENSE for details.


### **How to Use**:
1. Replace `[your-drive-link]` with your Google Drive dataset link.  
2. Add your Hugging Face Space link in `[your-hf-space-link]`.  
3. Include actual accuracy/loss values and training plots.  

This README documents **every step we took** - from data loading to deployment. Let me know if you want to add/remove anything! 🚀

