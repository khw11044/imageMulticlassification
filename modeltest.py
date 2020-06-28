from PIL import Image
import os, glob, numpy as np
import scipy
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
import matplotlib.pyplot as plt
from matplotlib.image import imread

caltech_dir = "./jpg"
image_w = 64
image_h = 64


X = []
filenames = []
files = glob.glob(caltech_dir+"/*.*")
for i, f in enumerate(files):
    img = Image.open(f)
    img = img.convert("RGB")
    img = img.resize((image_w, image_h))
    data = np.asarray(img)
    filenames.append(f)
    X.append(data)

X = np.array(X)
model = load_model('./model/multi_mushroom_classification.model')

prediction = model.predict(X)
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

cnt = 0
class_0_num = 0
class_1_num = 0
class_2_num = 0

for i in prediction:
    pre_ans = i.argmax()  # 예측 레이블
    #print("전체이미지개수{}".format(pre_ans))
    print(i)
    pre_ans_str = ''
    if pre_ans == 0: pre_ans_str = "붉은사슴뿔버섯"
    elif pre_ans == 1: pre_ans_str = "새송이버섯"
    elif pre_ans == 2: pre_ans_str = "표고버섯"

    if i[0] >= 0.8 : 
        print("해당 "+filenames[cnt].split("\\")[1]+"이미지는 "+pre_ans_str+"로 추정됩니다.")
        class_0_num = class_0_num + 1
    if i[1] >= 0.8: 
        print("해당 "+filenames[cnt].split("\\")[1]+"이미지는 "+pre_ans_str+"으로 추정됩니다.")
        class_1_num = class_1_num + 1
    if i[2] >= 0.8: 
        print("해당 "+filenames[cnt].split("\\")[1]+"이미지는 "+pre_ans_str+"로 추정됩니다.")
        class_2_num = class_2_num + 1
    cnt += 1

print("전체 버섯 {}개 중".format(class_0_num+class_1_num+class_2_num))
print("0클래스 : 붉은사슴뿔버섯은 {}개 입니다".format(class_0_num))
print("1클래스 : 새송이버섯은 {}개 입니다".format(class_1_num))
print("2클래스 : 표고버섯은 {}개 입니다".format(class_2_num))


last_weight = model.layers[-1].get_weights()[0]

new_model = Model(
    inputs = model.input,
    outputs=(
        model.layers[-5].output,
        model.layers[-1].output
    )
)
new_model.summary()
IMG_PATH = './jpg/'
imgnum = '표1.jpg'
test_img = img_to_array(load_img(os.path.join(IMG_PATH, imgnum), target_size=(64, 64)))
plt.imshow(test_img.astype(np.uint8))

test_input = preprocess_input(np.expand_dims(test_img.copy(), axis=0))

last_conv_output, pred = new_model.predict(test_input)

last_conv_output = np.squeeze(last_conv_output) #16, 16, 64
feature_activation_maps = scipy.ndimage.zoom(last_conv_output, (4, 4, 1), order=1) # (16, 16, 64) -> (64, 64, 64)
pred_class = np.argmax(pred)
predicted_class_weights = last_weight[:, pred_class]
print(predicted_class_weights.shape)