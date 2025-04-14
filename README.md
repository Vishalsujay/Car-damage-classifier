#  Car Damage Classifier

A deep learning-powered web application that automatically detects **car damage** and classifies its **location and severity** using a fine-tuned **ResNet-50** model. Built with **PyTorch** and deployed via **Streamlit**, this tool is ideal for automotive diagnostics, insurance claim automation, and car maintenance systems.

---

##  Features

- Detects car damage in images  
- Identifies if the damage is on the **front** or **rear**  
- Classifies severity into:  
  - Front Breakage  
  - Front Crushed  
  - Front Normal  
  - Rear Breakage  
  - Rear Crushed  
  - Rear Normal  
- Fine-tuned on top of **ResNet-50**  
- Instant predictions via a web UI using **Streamlit**  

---

##  Project Structure

```
car_damage_classifier/
├── Model_Training/
│   └── train_model.ipynb         # Notebook used to train the model
├── utils/
│   ├── predict.py                # Prediction and preprocessing logic
│   └── model.pth                 # Trained PyTorch model weights
├── app.py                        # Streamlit application entry point
├── requirements.txt              # Required Python dependencies
└── README.md                     # Project documentation
```

---

##  Getting Started

Follow the steps below to set up and run the project locally.

### Clone the Repository

```bash
git clone https://github.com/your-username/car_damage_classifier.git
cd car_damage_classifier
```

### Create and Activate a Virtual Environment (optional but recommended)

```bash
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
```

### Install Required Packages

```bash
pip install -r requirements.txt
```

###  Run the Streamlit App

```bash
streamlit run app.py
```

Once the app is running, a browser tab should open automatically. If not, copy the URL printed in the terminal and paste it into your browser.

---

##  Model Training Details

The model was trained using a custom dataset containing car images labeled with various types of damage. Training is done using a fine-tuned version of **ResNet-50**, where the last few layers were unfrozen and modified to output predictions for 6 damage categories.

You can find the training code in:

```
Model_Training/car_damage_detection.ipynb
```

The trained model is saved as:

```
utils/model.pth
```

---

##  Prediction Pipeline

The prediction logic is encapsulated in `utils/predict.py`:
- Loads and preprocesses the image
- Applies necessary transformations (resize, normalize, tensor conversion)
- Loads the trained model
- Runs inference and maps output to readable class labels

---

##  Use Cases

- Car insurance claim automation  
- Garage or workshop damage inspection  
- Car condition verification in rental services  
- Automotive visual diagnostics and record keeping  

---

##  Tech Stack

- Python 3.x  
- PyTorch  
- TorchVision  
- Streamlit  
- PIL (Pillow)  

---

##  To-Do

- [ ] Add drag-and-drop upload to Streamlit UI  
- [ ] Enable camera-based live prediction  
- [ ] Deploy app on Streamlit Cloud or HuggingFace Spaces  
- [ ] Improve model generalization with more diverse data  

---

## Sample Output
![image](https://github.com/user-attachments/assets/fc10dfd2-fde1-40ca-988e-3426e262f2f0)


---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).
