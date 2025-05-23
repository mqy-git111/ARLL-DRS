import json
import os

# 获取JSON文件的绝对路径
import random


# /media/user_gou/SSD/whole_body_seg/Data/nnUNet_raw/Dataset107_pelvic
file_path = os.path.abspath('./dataset/dataset_btcv.json')
old_path = os.path.abspath('./dataset/dataset.json')  #nnunet的json文件
train_imgpath = "./data/imagesTr"
train_labelpath = "./data/labelsTr"
test_imgpath = "./data/test/imagesTs"
test_labelpath = "./data/test/labelsTs"

# 读取JSON文件
def read_json_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
        return data


res = read_json_file(file_path)
old = read_json_file(old_path)
labels = dict()
old_label = old['labels']
# for i in range(len(old_label)):
for k,v in old_label.items():
    labels[v] = k
res['labels'] = labels

test_imgs = os.listdir(test_imgpath)
train_imgs = os.listdir(train_imgpath)
# train_imgs = [i for i in train_imgs_all if i[0] == 's']
nums = int(len(train_imgs) * 0.8)
random.shuffle(train_imgs)
train = train_imgs[:nums]
val = train_imgs[nums:]
# train = train_imgs[:24]
# val = train_imgs[-100:]
training = []
validation = []
test = []
temp = dict()
for test_img in test_imgs:
    temp = dict()
    temp['image'] = os.path.join("." + test_imgpath,test_img)
    temp['label'] = os.path.join("." + test_labelpath,test_img.replace("_0000.nii.gz",".nii.gz"))
    test.append(temp)
    # test.append(os.path.join(test_imgpath,test_img))

for img in train:
    temp = dict()
    temp['image'] = os.path.join("." + train_imgpath,img)
    temp['label'] = os.path.join("." + train_labelpath,img.replace("_0000.nii.gz",".nii.gz"))
    training.append(temp)

for img in val:
    temp = dict()
    temp['image'] = os.path.join("." + train_imgpath,img)
    temp['label'] = os.path.join("." + train_labelpath,img.replace("_0000.nii.gz",".nii.gz"))
    validation.append(temp)



res["training"] = training
res["validation"] = validation
res["test"] = test
res['numTest'] = len(test_imgs)
res["numTraining"] = len(train_imgs)
# res['numTest'] = len(test_imgs)
# res["numTraining"] = 524

# 将数据转换为JSON字符串
json_data = json.dumps(res)

# 打开文件
with open("./dataset/dataset_pelvic.json", "w") as file:
    
    # 写入JSON字符串到文件

    json.dump(res, file, indent=4)
print()
