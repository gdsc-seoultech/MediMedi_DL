from PIL import Image

img = Image.open('demo_image/demo.jpg')
f = open('result/res_demo.txt', 'r')
count = 0
while True:
    line = f.readline()
    if not line: break
    x = list(map(int,(line.strip().split(','))[0::2]))
    y = list(map(int,(line.strip().split(','))[1::2]))
    
    left = min(x)
    right = max(x)
    upper = min(y)
    lower = max(y)
    box = (left,upper,right,lower)
    count+=1
    
    img_crop = img.crop(box=box)
    img_crop.save("crop/img_crop_"+str(count)+".jpg")
f.close()