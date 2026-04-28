# 🕵️‍♂️ AI-Powered Deepfake & Image Manipulation Detection

An AI-based digital forensics tool designed to detect deepfakes and manipulated images.  
This system helps investigators analyze image authenticity, identify manipulation techniques, and generate forensic evidence reports.

---

## 🚨 Problem Statement

With the rise of deepfakes and AI-generated media, digital content can no longer be trusted blindly.  
Manipulated images are widely used in:

- Fraud & scams  
- Blackmail  
- Misinformation campaigns  

Investigators need reliable tools to determine whether an image is authentic or manipulated.

---

## 💡 Proposed Solution

This project provides a forensic investigation tool that:

- Detects deepfake faces using facial analysis  
- Identifies image splicing and tampering  
- Detects AI-generated content using statistical patterns  
- Analyzes compression artifacts (ELA & JPEG inconsistencies)  
- Generates a manipulation probability score  
- Produces a forensic evidence report  

---

## ⚙️ Technologies Used

- Python  
- TensorFlow  
- OpenCV  
- scikit-image  
- NumPy / Pandas  

---

## 📂 Project Structure

deepfake-forensics-tool/
│
├── data/                  
├── models/                
├── src/                   
│   ├── deepfake_detector.py
│   ├── splicing_detector.py
│   ├── ai_generated_detector.py
│   ├── compression_analysis.py
│   ├── report_generator.py
│
├── tests/                 
├── reports/               
├── app.py                 
├── requirements.txt
└── README.md

---

## 🚀 Installation

git clone https://github.com/your-username/deepfake-forensics-tool.git  
cd deepfake-forensics-tool  
pip install -r requirements.txt  

---

## ▶️ Usage

1. Add your image to the data/ folder  
2. Run the application:

python app.py  

3. The system will output a forensic report.

---

## 📊 Example Output

=== FORENSIC REPORT ===  
Image: data/test.jpg  
Deepfake Score: 0.50  
Splicing Score: 0.50  
AI Generated Score: 0.50  
Compression Score: 0.50  
Final Manipulation Probability: 0.50  

---

## 🧪 Testing & Validation

- Real images  
- Deepfake datasets  
- Manipulated images  

Metrics:
- Accuracy  
- Precision  
- Recall  

---

## 🔐 Forensic Considerations

- Image hashing  
- Evidence preservation  
- Chain of custody  

---

## 📌 Future Improvements

- Video deepfake detection  
- Streamlit dashboard  
- Explainable AI (SHAP)  
- Reverse image search  
- Visual highlighting  

---

## 📜 License

Educational use only.
