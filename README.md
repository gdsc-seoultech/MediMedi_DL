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

2. Text Recognition
```
python detect.py --source test_image/ --save-crop 
```
3. Text Detection
```
python recogn.py --image_folder detection/runs/detect/exp/crops/text/
```

