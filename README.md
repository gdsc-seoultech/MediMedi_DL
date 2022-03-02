"# MediMedi_DL"

## Getting Started
Clone repo and install requirements.txt in a Python>=3.7.0 environment
```
git clone https://github.com/gdsc-seoultech/MediMedi_DL.git
cd MediMedi_DL
pip install -r requirements.txt
```
1. Image를 받으면 demo_image 혹은 test_image 폴더에 저장

2. Text Recognition
```
#demo
python detect.py --trained_model=craft_mlt_25k.pth --test_folder=demo_image/
#test
python detection/detect.py --weights detection/best.pt --source test_image/ --save-crop 
```
3. Image Crop(test에서는 건너뛰어도 되는 과정)
```
python crop.py
```
4. Text Detection
```
#demo
python recognize.py \
--Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction Attn \
--image_folder crop/ \
--saved_model TPS-ResNet-BiLSTM-Attn.pth
#test
python recognition/recogn.py 
--Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction Attn \
--image_folder detection/runs/detect/exp/crops/text/ --saved_model recognition/best_accuracy.pth \
--imgH 64 --imgW 200
```

