# yolo_auto_agmentation

## 소개
`yolo_auto_agmentation`은 YOLO 형식의 데이터셋을 자동으로 증강 및 시각화할 수 있는 Python 패키지입니다. 이 패키지는 다양한 이미지 증강 기법을 사용하여 데이터셋을 풍부하게 만들고, 증강된 이미지를 시각화하여 확인할 수 있는 기능을 제공합니다.

## 설치

### pip를 통한 설치
PyPI에 배포된 패키지를 설치하려면 다음 명령어를 사용하세요:
```bash
pip install yolo-auto-agmentation
```

## requirements.txt를 통한 설치
```bash
pip install -r requirements.txt
```

`requirements.txt`의 내용:
```bash
opencv-python
tqdm
albumentations
natsort
```

## 사용법
1. 자동 증강 기능
`auto_augment` 메서드를 사용하여 YOLO 형식의 데이터셋을 자동으로 증강할 수 있습니다. 다음은 사용 예시입니다:

```python
from yolo_auto_agment import Img_aug

img_aug = Img_aug()
img_aug.auto_augment(dataset_path='test_dataset', repeat=3)

```

위 코드는 `test_dataset` 폴더의 데이터를 세 번 반복하여 증강합니다. 증강된 데이터는 `test_dataset_aug` 폴더에 저장됩니다.

2. 증강된 이미지 시각화
`auto_draw` 메서드를 사용하여 증강된 이미지의 바운딩 박스를 시각화할 수 있습니다. 다음은 사용 예시입니다:

```python
from yolo_auto_agment import Img_aug

img_aug = Img_aug()
img_aug.auto_draw(auged_path='test_dataset_aug')
```

위 코드는 `test_dataset_aug` 폴더에 저장된 증강된 이미지에 바운딩 박스를 그려 `test_dataset_aug_draw` 폴더에 저장합니다.

## 주요 클래스 및 메서드

### Img_aug 클래스

`Img_aug` 클래스는 다양한 이미지 증강 기법을 사용하여 데이터를 증강하고, 이를 시각화하는 기능을 제공합니다.

#### 초기화 메서드
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

#### auto_augment 메서드
```python
auto_augment(dataset_path, repeat=1)
```

- `dataset_path`: YOLO 형식의 train, val, test 데이터셋 폴더 경로 
- `repeat`: 데이터셋을 증강할 횟수 (기본값: 1)

#### auto_draw 메서드
```python
auto_draw(auged_path, rand=True, ea=1000)
```

- `auged_path`: 증강된 데이터셋 폴더 경로
- `rand`: 이미지를 랜덤으로 셔플할지 여부 (기본값: True)
- `ea`: 시각화할 이미지 수 (기본값: 1000)


## 예제 코드
다음은 yolo_auto_agmentation 패키지를 사용하는 간단한 예제 코드입니다:

```python
from yolo_auto_agment import Img_aug

# 인스턴스 생성
img_aug = Img_aug()

# 데이터 증강
img_aug.auto_augment(dataset_path='test_dataset', repeat=3)

# 증강된 이미지 시각화
img_aug.auto_draw(auged_path='test_dataset_aug')
```

## 기여
기여를 원하신다면 GitHub 저장소를 포크하고 풀 리퀘스트를 제출해 주세요. 버그 리포트와 기능 요청도 환영합니다.

## 라이선스
이 프로젝트는 MIT 라이선스 하에 배포됩니다. 자세한 내용은 LICENSE 파일을 참조하세요.