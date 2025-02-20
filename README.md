# Email Purchase Prediction  

This project uses **Natural Language Processing (NLP)** and **Machine Learning** to predict whether an email indicates interest in purchasing a product. It utilizes **TF-IDF vectorization** and a **Random Forest Classifier** for classification.  

## Features  
- Cleans and preprocesses email text  
- Uses **TF-IDF (Term Frequency - Inverse Document Frequency)** for feature extraction  
- Implements a **Random Forest Classifier** for predictions  
- Provides a simple function to predict purchase intent based on email text  

## Installation  

```bash
git clone https://github.com/YOUR_GITHUB_USERNAME/Email-Purchase-Predictor.git
cd Email-Purchase-Predictor
pip install pandas numpy scikit-learn
```  

## Usage  

```python
from email_predictor import predict_purchase

email_text = "I am interested in your product. Can you share pricing?"
print(f"Prediction: {predict_purchase(email_text)}")
```  

**Output:**  
```bash
Prediction: Likely to Buy
```  

## How It Works  
- **Predefined Training Data:** The model is trained on a small sample of past emails.  
- **Text Cleaning:** The email text is converted to lowercase and special characters are removed.  
- **TF-IDF Vectorization:** Converts text into numerical features.  
- **Random Forest Classification:** Predicts if the email suggests a purchase.  

## Customization  
To train on your own dataset, modify the `data` dictionary inside `train_model()` in `email_predictor.py`.  

```python
data = {
    "email": [
        "Is there a free trial?",
        "Thanks, I’ll place an order today.",
        "Not interested at this time."
    ],
    "bought": [0, 1, 0]
}
```  

Author:Harish
Intern:Minervasoft
