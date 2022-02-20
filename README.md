"# MediMedi_DL" 
1. Image를 받으면 demo_image 폴더에 저장

2. Text Recognition
```
python detect.py --trained_model=craft_mlt_25k.pth --test_folder=demo_image/
```
3. Image Crop
```
python crop.py
```
4. Text Detection
```
python3 recognize.py \
--Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction Attn \
--image_folder crop/ \
--saved_model TPS-ResNet-BiLSTM-Attn.pth
```
