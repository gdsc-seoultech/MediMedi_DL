"# MediMedi_DL" 

```
python detect.py --trained_model=craft_mlt_25k.pth --test_folder=demo_image/
```

```
python crop.py
```

```
python3 recognize.py \
--Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction Attn \
--image_folder crop/ \
--saved_model TPS-ResNet-BiLSTM-Attn.pth
```
