# yolo_auto_agmentation

## 소개 (Introduction)
`yolo_auto_agmentation`은 YOLO 형식의 데이터셋을 자동으로 증강 및 시각화할 수 있는 Python 패키지입니다. 이 패키지는 다양한 이미지 증강 기법을 사용하여 데이터셋을 풍부하게 만들고, 증강된 이미지를 시각화하여 확인할 수 있는 기능을 제공합니다.

`yolo_auto_agmentation` is a Python package that allows automatic augmentation and visualization of datasets in YOLO format. This package uses various image augmentation techniques to enrich the dataset and provides functionality to visualize augmented images.

## 설치 (Installation)

### pip를 통한 설치 (Install via pip)
PyPI에 배포된 패키지를 설치하려면 다음 명령어를 사용하세요:
To install the package published on PyPI, use the following command:
```bash
pip install yolo-auto-agmentation
```

## requirements.txt를 통한 설치 (Install via requirements.txt)
```bash
pip install -r requirements.txt
```

`requirements.txt`의 내용(Content of requirements.txt):
```bash
opencv-python
tqdm
albumentations
natsort
```

## 사용법 (Usage)
1. 자동 증강 기능 (Auto Augmentation Function)
`auto_augment` 메서드를 사용하여 YOLO 형식의 데이터셋을 자동으로 증강할 수 있습니다. 다음은 사용 예시입니다:
You can automatically augment a dataset in YOLO format using the auto_augment method. Here is an example:

```python
from yolo_auto_agment import Img_aug

img_aug = Img_aug()
img_aug.auto_augment(dataset_path='test_dataset', repeat=3)

```

위 코드는 `test_dataset` 폴더의 데이터를 세 번 반복하여 증강합니다. 증강된 데이터는 `test_dataset_aug` 폴더에 저장됩니다.
The above code augments the data in the `test_dataset folder` three times. The augmented data is saved in the `test_dataset_aug` folder.

2. 증강된 이미지 시각화 (Visualize Augmented Images)
`auto_draw` 메서드를 사용하여 증강된 이미지의 바운딩 박스를 시각화할 수 있습니다. 다음은 사용 예시입니다:
You can visualize the bounding boxes of augmented images using the `auto_draw` method. Here is an example:

```python
from yolo_auto_agment import Img_aug

img_aug = Img_aug()
img_aug.auto_draw(auged_path='test_dataset_aug')
```

위 코드는 `test_dataset_aug` 폴더에 저장된 증강된 이미지에 바운딩 박스를 그려 `test_dataset_aug_draw` 폴더에 저장합니다.
The above code draws bounding boxes on the augmented images stored in the `test_dataset_aug` folder and saves them in the `test_dataset_aug_draw` folder.

## 데이터셋 구조 (Dataset Structure)
올바른 데이터셋 구조는 다음과 같아야 합니다.
The correct dataset structure should be as follows:

```css
dataset_path
  ├── train
  │   ├── images
  │   │   ├── a.png
  │   │   ├── b.png
  │   │   └── ...
  │   └── labels
  │       ├── a.txt
  │       ├── b.txt
  │       └── ...
  ├── val
  │   ├── images
  │   │   ├── c.png
  │   │   └── ...
  │   └── labels
  │       ├── c.txt
  │       └── ...
  └── test
      ├── images
      │   ├── d.png
      │   └── ...
      └── labels
          ├── d.txt
          └── ...
```

- `dataset_path`: 데이터셋의 최상위 경로입니다.

- `train`, `val`, `test`: 각 데이터셋 폴더는 학습, 검증, 테스트 데이터셋을 포함합니다.

- `images` 폴더: 이미지 파일(.png, .jpg 등)을 포함합니다.

- `labels` 폴더: 각 이미지에 해당하는 YOLO 형식의 레이블 파일(.txt)을 포함합니다.

- `dataset_path`: The top-level path of the dataset.

- `train`, `val`, `test`: Each dataset folder contains training, validation, and test datasets respectively.

- `images` folder: Contains image files (e.g., .png, .jpg).

- `labels` folder: Contains YOLO format label files (e.g., .txt) corresponding to each image

이 구조에 따라 데이터를 구성하면 `yolo_auto_agmentation` 패키지를 사용하여 데이터 증강 및 시각화를 쉽게 수행할 수 있습니다.
If you organize your data according to this structure, you can easily perform data augmentation and visualization using the `yolo_auto_agmentation` package.



## 주요 클래스 및 메서드 (Main Class and Methods)

### Img_aug 클래스 (Img_aug Class)

`Img_aug` 클래스는 다양한 이미지 증강 기법을 사용하여 데이터를 증강하고, 이를 시각화하는 기능을 제공합니다.
The `Img_aug` class provides functionality to augment data using various image augmentation techniques and to visualize them.

#### 초기화 메서드 (Initialization Method)
```python
Img_aug(
    horizontalflip_p=0.5,
    rotate_limit=20, rotate_p=0.5,
    colorjitter_brightness=0.5, colorjitter_contrast=0.5, colorjitter_saturation=0.3, colorjitter_hue=0.15, colorjitter_p=0.5,
    gaussianblur_limit_from=3, gaussianblur_limit_to=7, gaussianblur_p=0.5,
    motionblur_limit_from=3, motionblur_limit_to=7, motionblur_p=0.5,
    randomcrop_p=0.5
)
```

각 매개변수는 증강 기법의 확률과 한계를 설정합니다.
Each parameter sets the probability and limits of the augmentation techniques.

#### Custom Random Crop 메서드 (Custom Random Crop Method)
이 레포지토리의 핵심 기술이라고 할 수 있는 Albumentation에 없는 Custom 제작된 메서드 입니다. 이미지 상에 있는 ground truth인 bounding box를 지키면서 랜덤하게 crop합니다. 아래는 이에 대한 상세 설명입니다.
This is a custom-made method that is not available in Albumentation and can be considered a core technology of this repository. It randomly crops while preserving the bounding box, which is the ground truth on the image. Below is a detailed explanation.

![alt text](https://raw.githubusercontent.com/Nyan-SouthKorea/YOLO_Auto_Agmentation/main/readme_img/image1.png)


<상세 설명> (Detailed Explanation)
YOLO형식의 img, label path를 입력받아 증강 후 원하는 path에 저장해준다.
It takes the YOLO format img, label path, augments them, and saves them to the desired path.


알고리즘 진행 순서: (Algorithm Procedure)
1. 고화질 이미지, 레이블 읽어오기 (Read high-quality images and labels) (Read high-quality images and labels)
2. random crop 진행(bbox를 해치지 않는 선에서) (Perform random crop while preserving bbox)
3. 나머지 빠른 증강을 위해 작은 이미지로 resize(640) (Resize to 640 for faster remaining augmentation)
4. 나머지 증강 효과 적용 (Apply remaining augmentation effects)
5. 저장 (Save)

이렇게 수행하는 이유는 아래와 같다 (The reasons for this procedure are as follows:)
- 랜덤 크롭 시 화질 저하를 막기 위해서 (To prevent quality degradation during random cropping)
- 나머지 증강시, 원본 이미지로 진행 시 시간이 너무 오래걸림 (To avoid excessive processing time when performing the remaining augmentation on the original image)

```python
def random_crop(self, img, bboxes, p=0.5):
    '''
    bbox 해치지 않는 선에서 랜덤 crop하여 반환
    
    img: cv2.imread()로 읽은 이미지
    bboxes: txt_to_bbox()로 변환한 Albumentations 형식의 bbox(YOLO랑 순서 다름)
    p: 랜덤 crop이 적용될 확률
    '''
    # 적용된 확률로 crop 진행할지 말지 판단
    if random() < p:
        h, w, c = img.shape
        # Crop할 수 있는 가장 큰 박스 찾기
        x1_list, y1_list, x2_list, y2_list = [], [], [], []
        for bbox in bboxes:
            b1, b2, b3, b4, class_no = bbox
            x1, y1, x2, y2 = self.b1b2b3b4_to_x1y1x2y2(b1, b2, b3, b4)
            x1_list.append(x1)
            y1_list.append(y1)
            x2_list.append(x2)
            y2_list.append(y2)
        if len(bboxes) == 0: # bbox가 없을 경우(중앙에 콩알만한 박스까지 crop 가능)
            x1 = int((w/2)-1)
            y1 = int((h/2)-1)
            x2 = x1 + 2
            y2 = y1 +2
            max_crop = {'x1':x1, 'y1':y1, 'x2':x2, 'y2':y2}
        else: # bbox가 있을 경우
            x1_min = min(x1_list)
            y1_min = min(y1_list)
            x2_max = max(x2_list)
            y2_max = max(y2_list)
            x1_min, y1_min, x2_max, y2_max = self.nor_bbox_to_pixel_bbox(x1_min, y1_min, x2_max, y2_max, h, w) # 픽셀값 변환
            max_crop = {'x1':x1_min, 'y1':y1_min, 'x2':x2_max, 'y2':y2_max}
        # crop 시작
        crop_x1 = randint(0, max_crop['x1'])
        crop_y1 = randint(0, max_crop['y1'])
        crop_x2 = randint(max_crop['x2'], w)
        crop_y2 = randint(max_crop['y2'], h)
        # img crop
        crop_img = img[crop_y1:crop_y2, crop_x1:crop_x2]
        h_, w_, c_ = crop_img.shape
        # label crop
        crop_bboxes = []
        for bbox in bboxes:
            # x1, y1, x2, y2 형식으로 변경
            b1, b2, b3, b4, class_no = bbox
            x1, y1, x2, y2 = self.b1b2b3b4_to_x1y1x2y2(b1, b2, b3, b4)
            x1, y1, x2, y2 = self.nor_bbox_to_pixel_bbox(x1, y1, x2, y2, h, w)
            # crop된 영역 bbox에 적용(left top부분만 영향을 끼친다)
            x1 -= crop_x1
            y1 -= crop_y1
            x2 -= crop_x1
            y2 -= crop_y1
            # 다시 정규화된 b1, b2, b3, b4 형식으로 변경
            x1, y1, x2, y2 = self.pixel_bbox_to_nor_bbox(x1, y1, x2, y2, h_, w_)
            b1, b2, b3, b4 = self.x1y1x2y2_to_b1b2b3b4(x1, y1, x2, y2)
            crop_bboxes.append([b1, b2, b3, b4, class_no])
        return crop_img, crop_bboxes
    else:
        return img, bboxes
```




#### auto_augment 메서드 (auto_augment Method)
```python
auto_augment(dataset_path, repeat=1)
```

- `dataset_path`: YOLO 형식의 train, val, test 데이터셋 폴더 경로 
- `repeat`: 데이터셋을 증강할 횟수 (기본값: 1)
- `dataset_path`: Path to the train, val, and test dataset folders in YOLO format.
- `repeat`: Number of times to augment the dataset (default: 1).

#### auto_draw 메서드 (auto_draw Method)
```python
auto_draw(auged_path, rand=True, ea=1000)
```

- `auged_path`: 증강된 데이터셋 폴더 경로
- `rand`: 이미지를 랜덤으로 셔플할지 여부 (기본값: True)
- `ea`: 시각화할 이미지 수 (기본값: 1000)
- `auged_path`: Path to the augmented dataset folder.
- `rand`: Whether to shuffle the images randomly (default: True).
- `ea`: Number of images to visualize (default: 1000).


## 예제 코드 (Example Code)
다음은 yolo_auto_agmentation 패키지를 사용하는 간단한 예제 코드입니다:
Here is a simple example code using the yolo_auto_agmentation package:

```python
from yolo_auto_agment import Img_aug

# 인스턴스 생성
img_aug = Img_aug()

# 데이터 증강
img_aug.auto_augment(dataset_path='test_dataset', repeat=3)

# 증강된 이미지 시각화
img_aug.auto_draw(auged_path='test_dataset_aug')
```

## 기여 (Contributing)
기여를 원하신다면 GitHub 저장소를 포크하고 풀 리퀘스트를 제출해 주세요. 버그 리포트와 기능 요청도 환영합니다.
If you would like to contribute, please fork the GitHub repository and submit a pull request. Bug reports and feature requests are also welcome.

## 라이선스 (License)
이 프로젝트는 MIT 라이선스 하에 배포됩니다. 자세한 내용은 LICENSE 파일을 참조하세요.
This project is distributed under the MIT License. See the LICENSE file for more details.

## Github 주소 (Github URL)
https://github.com/Nyan-SouthKorea/YOLO_Auto_Agmentation

## Pypi 주소 (PyPI URL)
https://pypi.org/project/yolo-auto-agmentation/ 

## 네이버 블로그 주소 (Naver Blog URL)
https://blog.naver.com/112fkdldjs 