# MediMedi_DL

## Getting Started
Clone repo and install requirements.txt in a Python>=3.7.0 environment
```
git clone https://github.com/gdsc-seoultech/MediMedi_DL.git
cd MediMedi_DL
pip install -r requirements.txt
```
## Running OCR Code
1. Image를 test_image 폴더에 저장

2. Text Detection
```
python detect.py --source test_image/
```
3. Text Recognition
```
python recogn.py --image_folder runs/detect/exp/
```

