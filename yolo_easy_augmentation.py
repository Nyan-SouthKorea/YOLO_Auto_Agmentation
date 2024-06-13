# 기본 라이브러리
import os
from random import randint, random, shuffle
import time
import shutil

# 추가 라이브러리
import cv2
from tqdm import tqdm
import albumentations as A
from natsort import natsorted

class Img_aug:
    def __init__(self, horizontalflip_p=0.5,
                 rotate_limit=20, rotate_p=0.5,
                 colorjitter_brightness=0.5, colorjitter_contrast=0.5, colorjitter_saturation=0.3, colorjitter_hue=0.15, colorjitter_p=0.5,
                 gaussianblur_limit_from=3, gaussianblur_limit_to=7, gaussianblur_p=0.5,
                 motionblur_limit_from=3, motionblur_limit_to=7, motionblur_p=0.5,
                 randomcrop_p=0.5):
        self.bbox_params = A.BboxParams(format='yolo')
        self.spent_time = 0
        # 호리젠탈플립
        self.horizontalflip_p = horizontalflip_p
        # 회전
        self.rotate_limit = rotate_limit
        self.rotate_p = rotate_p
        # 색감
        self.colorjitter_brightness = colorjitter_brightness
        self.colorjitter_contrast = colorjitter_contrast
        self.colorjitter_saturation = colorjitter_saturation
        self.colorjitter_hue =colorjitter_hue
        self.colorjitter_p = colorjitter_p
        # 가우시안 블러
        self.gaussianblur_limit_from = gaussianblur_limit_from
        self.gaussianblur_limit_to = gaussianblur_limit_to
        self.gaussianblur_p =gaussianblur_p
        # 모션블러
        self.motionblur_limit_from =motionblur_limit_from
        self.motionblur_limit_to = motionblur_limit_to
        self.motionblur_p = motionblur_p
        # 랜덤 크롭
        self.randomcrop_p =randomcrop_p


    def auto_augment(self, dataset_path, repeat=1):
        '''
        dataset 경로를 넣으면 자동으로 증강됨
        
        dataset_path: YOLO형식의 train, val, test가 존재하는 폴더(test는 꼭 있을 필요 없음)
        반드시 지켜져야 하는 파일 경로 형식
        데이터셋 폴더
          ㄴ train
          ㄴ val
          ㄴ test
          (train, val, test 중 하나 이상만 존재하면 되지만, 꼭 폴더가 있어야 함)
        repeat: 데이터셋을 증강할 횟수. 1로 설정할 경우 원본 + 증강 1번에 대한 데이터셋이 새로 저장됨
        '''
        # 파일 유효성 검사
        if self.is_file_valid(dataset_path) == False:
            print('파일 유효성 검사 문제 발생')
            return
        else:
            print('파일 유효성 검사 문제 없음')
            
        repeat += 1 # 처음은 자기 자신 복사
        # 증강 폴더 및 경로 생성
        if dataset_path[-1] == '/': # 경로 마지막에 /가 붙으면 에러가 나기 때문
            dataset_path = dataset_path[:-1]
        aug_path = f'{dataset_path}_aug'
        # 증강 전체 flow 시작
        folder_list = os.listdir(dataset_path)
        for aug_cnt in range(repeat):
            for mode in ['train', 'val', 'test']:
                if not mode in folder_list: continue # 폴더가 없으면 패스
                # 폴더 생성
                for data in ['images', 'labels']:
                    os.makedirs(f'{aug_path}/{mode}/{data}', exist_ok=True)
                # 이미지 리스트 불러오기
                for img_name in tqdm(os.listdir(f'{dataset_path}/{mode}/images'), desc=f'{mode}, {aug_cnt}'):
                    if self.img_format(img_name) == False: continue # 이미지 포맷이 아닌 경우 pass
                    label_name = f'{img_name.split(".")[0]}.txt'
                    # 필요 경로 설정
                    r_img_path = f'{dataset_path}/{mode}/images/{img_name}'
                    r_label_path = f'{dataset_path}/{mode}/labels/{label_name}'
                    # 처음에는 자기 자신 복사하기(리사이즈는 적용)
                    if aug_cnt == 0:
                        w_img_path = f'{aug_path}/{mode}/images/{img_name}'
                        w_label_path = f'{aug_path}/{mode}/labels/{label_name}'
                        # 원본 이미지, 레이블 복사
                        resized_img = self.resize(cv2.imread(r_img_path))
                        cv2.imwrite(w_img_path, resized_img)
                        shutil.copy(r_label_path, w_label_path)
                    # 증강 시작
                    else:
                        # path 정의하기
                        name_without_format = img_name.split('.')[0]
                        aug_img_name = f'{name_without_format}_aug{aug_cnt}.png'
                        aug_label_name = f'{name_without_format}_aug{aug_cnt}.txt'
                        w_img_path = f'{aug_path}/{mode}/images/{aug_img_name}'
                        w_label_path = f'{aug_path}/{mode}/labels/{aug_label_name}'
                        # 증강
                        self.img_aug(r_img_path, r_label_path, w_img_path, w_label_path)


    def auto_draw(self, auged_path, rand=True, ea=1000):
        '''
        증강된 사진 경로를 입력하면, bbox를 그려서 저장해준다
        auged_path: 증강된 데이터셋의 폴더 경로
        rand: 랜덤으로 셔플할 것인지, natsort로 할 것인지
        ea: 몇 장 랜덤으로 그려볼 것인지
        '''
        # 폴더 및 경로 생성
        if auged_path[-1] == '/': # 경로 마지막에 /가 붙으면 에러가 나기 때문
            auged_path = auged_path[:-1]
        draw_path = f'{auged_path}_draw'

        # 전체 그리기 flow 시작
        folder_list = os.listdir(auged_path)
        for mode in ['train', 'val', 'test']:
            if not mode in folder_list: continue # 폴더가 없으면 패스
            # 폴더 생성
            os.makedirs(f'{draw_path}/{mode}', exist_ok=True)
            # 이미지 리스트 불러오기
            img_list = os.listdir(f'{auged_path}/{mode}/images')
            # 정렬 옵션
            if rand:
                shuffle(img_list)
            else:
                img_list = natsorted(img_list)
            # 이미지 한 장씩 불러오기
            for i, img_name in enumerate(tqdm(img_list, desc=f'{mode}')):
                try:
                    label_name = f'{img_name.split(".")[0]}.txt'
                    if i >= ea: break
                    # 그리기 전 경로 설정
                    r_img_path = f'{auged_path}/{mode}/images/{img_name}'
                    r_label_path = f'{auged_path}/{mode}/labels/{label_name}'
                    w_img_path = f'{draw_path}/{mode}/{img_name}'
                    self.draw(r_img_path, r_label_path, w_img_path)
                except Exception as e:
                    print(f'에러가 발생했지만, 계속 진행: {e}')
                    continue
            

    def img_aug(self, r_img_path, r_label_path, w_img_path, w_label_path):
        '''
        YOLO형식의 img, label path를 입력받아 증강 후 원하는 path에 저장해준다.
        알고리즘 진행 순서:
        1. 고화질 이미지, 레이블 읽어오기
        2. random crop 진행(bbox를 해치지 않는 선에서)
        3. 나머지 빠른 증강을 위해 작은 이미지로 resize(640)
        4. 나머지 증강 효과 적용
        5. 저장

        이렇게 수행하는 이유는 아래와 같다
           - 랜덤 크롭 시 화질 저하를 막기 위해서
           - 나머지 증강시, 원본 이미지로 진행 시 시간이 너무 오래걸림

        r_img_path: 읽어오는 이미지 경로
        r_label_path: 읽어오는 레이블 경로
        w_img_path: 쓰고싶은 이미지 경로
        w_label_path: 쓰고싶은 레이블 경로
        '''
        start_time = time.time()
        # 이미지, 레이블 읽어오기
        img = cv2.imread(r_img_path)
        bboxes = self.txt_to_bbox(r_label_path)

        # bbox를 해치지 않는 선에서 random crop 진행(resize 병행)
        img, bboxes = self.random_crop(img, bboxes, p=self.randomcrop_p)

        # 화질 적당하기 줄이기
        img = self.resize(img)

        # 나머지 증강 옵션 설정
        transform = A.Compose([
            A.HorizontalFlip(p=self.horizontalflip_p), # 이미지 랜덤 호리젠탈 플립
            A.Rotate(limit=self.rotate_limit, p=self.rotate_p), # 이미지 회전 (-n도부터 n도 사이에서 무작위로 선택)
            A.ColorJitter(brightness=self.colorjitter_brightness, contrast=self.colorjitter_contrast, 
                          saturation=self.colorjitter_saturation, hue=self.colorjitter_hue, p=self.colorjitter_p),# 이미지의 색상 변화
            A.GaussianBlur(blur_limit=(self.gaussianblur_limit_from, self.gaussianblur_limit_to), p=self.gaussianblur_p), # 이미지에 가우시안 블러 적용
            A.MotionBlur(blur_limit=(self.motionblur_limit_from, self.motionblur_limit_to), p=self.motionblur_p) # 이미지에 모션 블러 적용
        ], bbox_params = self.bbox_params)

        # 어그멘테이션 실행
        transformed = transform(image = img, bboxes = bboxes)
        aug_img = transformed['image']
        aug_bboxes = transformed['bboxes']

        # 증강된 이미지, 레이블 저장
        cv2.imwrite(w_img_path, aug_img)
        self.bbox_to_txt(aug_bboxes, w_label_path)
        self.spent_time = time.time()-start_time


    def draw(self, r_img_path, r_label_path, w_img_path):
        '''
        언제든지 증강된 결과물을 그려볼 수 있는 기능

        r_img_path: 입력 이미지 경로
        r_label_path: 입력 레이블 경로
        w_img_path: 그린 이미지 출력 경로
        '''
        # 이미지 읽기
        img = cv2.imread(r_img_path)
        h, w, c = img.shape
        # 레이블 읽기
        bboxes = self.txt_to_bbox(r_label_path) # Albumentation 형식 bbox
        # 그리기
        for bbox in bboxes:
            b1, b2, b3, b4, class_no = bbox
            x1, y1, x2, y2 = self.b1b2b3b4_to_x1y1x2y2(b1, b2, b3, b4)
            x1, y1, x2, y2 = self.nor_bbox_to_pixel_bbox(x1, y1, x2, y2, h, w)
            cv2.rectangle(img, (x1,y1), (x2,y2), (0,0,255), 2)
            cv2.putText(img, str(class_no), (x1+3,y1+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
        cv2.imwrite(w_img_path, img)


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


    def bbox_fix(self, b1, b2, b3, b4):
        '''
        bbox가 이미지 밖으로 나가지 않도록 보정

        b1, b2, b3, b4: YOLO형식 순서로 center_x, center_y, x_len, y_len이 0 ~ 1로 정규화 된 값
        '''
        # 형식 변환
        x1, y1, x2, y2 = self.b1b2b3b4_to_x1y1x2y2(b1, b2, b3, b4)
        # 사진 밖으로 나갈 수 없도록 보정
        x1, y1, x2, y2 = max(0, x1), max(0, y1), max(0, x2), max(0, y2)
        x1, y1, x2, y2 = min(1, x1), min(1, y1), min(1, x2), min(1, y2)
        x2, y2 = max(x1, x2), max(y1, y2) # x2, y2가 최소한 x1, y2보다 작은 현상 방지
        # 형식 변환
        b1, b2, b3, b4 = self.x1y1x2y2_to_b1b2b3b4(x1, y1, x2, y2)
        return b1, b2, b3, b4


    def txt_to_bbox(self, r_label_path):
        '''
        YOLO 형식의 레이블을 Albumentation bbox로 반환

        r_label_path: YOLO 레이블 경로
        반환값: center_x, center_y, x_len, y_len, class_no 형식으로 반환(class_no가 뒤로 감. YOLO형식이 아니므로 주의!)
        '''
        # 레이블이 없는 경우 비어있는 리스트 반환
        if os.path.exists(r_label_path) == False:
            print(f'레이블 없음. 이미지만 증강되고 빈 레이블 생성: {r_label_path}')
            return []
        with open(r_label_path, 'r') as f:
            full_txt = f.read()
        txt_list = full_txt.split('\n')
        # 애초에 레이블링 되지 않은 이미지인지 판별
        bboxes = []
        for txt in txt_list:
            if len(txt) == 0: break
            bbox = txt.split(' ')
            if len(bbox) == 5: 
                b1, b2, b3, b4 = self.bbox_fix(bbox[1], bbox[2], bbox[3], bbox[4]) # bbox가 이미지 밖으로 나가지 않도록 수정
                bboxes.append([b1, b2, b3, b4, int(bbox[0])])        
        return bboxes


    def bbox_to_txt(self, bboxes, w_label_path):
        '''
        Albumentation형식의 bbox를 YOLO 레이블로 txt저장

        bboxes: Albumentation형식의 bbox [center_x, center_y, x_len, y_len, class_no]
        w_label_path: YOLO 레이블 txt파일 저장 경로
        '''
        with open(w_label_path, 'w') as f:
            f.write('') # 비어있는 텍스트 생성(bbox에 내용이 없을 경우 비어있는 레이블 텍스트 생성해야 함)
            # bbox 사진 밖으로 나가지 않도록 수정하여 실시간 저장
            for idx, bbox in enumerate(bboxes):
                b1, b2, b3, b4 = self.bbox_fix(bbox[0], bbox[1], bbox[2], bbox[3])
                fixed_bbox = [b1, b2, b3, b4, bbox[4]]
                # 엔터 조건 설정
                if idx == len(bboxes)-1: enter = ''
                else: enter = '\n'
                # 한줄씩 레이블 정보 txt로 작성
                f.write(f'{int(bbox[4])} {round(bbox[0], 5)} {round(bbox[1], 5)} {round(bbox[2], 5)} {round(bbox[3], 5)}{enter}')


    def x1y1x2y2_to_b1b2b3b4(self, x1, y1, x2, y2):
        '''
        정규화된 값이니 착각하지 말도록
        x1, y1, x2, y2: left top xy가 1 / right bottom xy가 2
        b1, b2, b3, b4: center_x, center_y, x_len, y_len
        '''
        x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
        x_len = x2-x1
        y_len = y2-y1
        center_x = (x1+x2)/2
        center_y = (y1+y2)/2
        b1, b2, b3, b4 = center_x, center_y, x_len, y_len
        return b1, b2, b3, b4
    

    def b1b2b3b4_to_x1y1x2y2(self, b1, b2, b3, b4):
        '''
        설명은 x1y1x2y2_to_b1b2b3b4() 함수 참조
        '''
        b1, b2, b3, b4 = float(b1), float(b2), float(b3), float(b4)
        center_x, center_y, x_len, y_len = b1, b2, b3, b4
        x1 = center_x - (x_len/2)
        y1 = center_y - (y_len/2)
        x2 = x1 + x_len
        y2 = y1 + y_len
        return x1, y1, x2, y2
        

    def resize(self, img, size_w=640):
        '''
        비율을 해치지 않고 resize하여 반환

        img: cv2 이미지
        '''
        h, w, c = img.shape
        return cv2.resize(img, (size_w, int(h/w*size_w)))


    def nor_bbox_to_pixel_bbox(self, x1, y1, x2, y2, h, w):
        '''
        정규화된 bbox 좌표를 넣으면, 픽셀값 bbox를 반환해준다
        x1, y1, x2, y2: bbox 좌표(0~1사이 정규화된 값)
        h, w: img.shape하면 나오는 h, w, c 값 중 c값을 제외하고 넣는다
        '''
        x1, y1, x2, y2 = int(x1*w), int(y1*h), int(x2*w), int(y2*h)
        x1, y1, x2, y2 = max(0,x1), max(0, y1), max(0, x2), max(0, y2)
        x1, y1, x2, y2 = min(w,x1), min(h,y1), min(w,x2), min(h,y2)
        return x1, y1, x2, y2
    

    def pixel_bbox_to_nor_bbox(self, x1, y1, x2, y2, h, w):
        '''
        픽셀값의 bbox 좌표를 넣으면, 정규화된 bbox를 반환해준다
        x1, y1, x2, y2: bbox 좌표(pixel 값)
        h, w: img.shape하면 나오는 h, w, c 값 중 c값을 제외하고 넣는다
        '''
        return x1/w, y1/h, x2/w, y2/h
    

    def is_file_valid(self, dataset_path, fix=True):
        '''
        데이터셋이 전체적으로 문제 없는 유효한 파일 구성인지 확인한다. 아래는 올바른 dataset 구성 예시
        dataset_path
        ㄴ train
           ㄴ images
              ㄴ a.png
              ㄴ b.png
                 ...
           ㄴ labels
              ㄴ a.txt
              ㄴ b.txt
                 ...
        ㄴ val
           ...
        ㄴ test
           ...
        
        dataset_path: 데이터셋 경로
        fix: 이미지에 매칭되는 레이블이 없을 경우 빈 레이블 추가할지 옵션
        '''
        try:
            folder_list = os.listdir(dataset_path)
            for mode in ['train', 'val', 'test']:
                if not mode in folder_list: continue
                # 이미지와 레이블 매칭 검사
                for img_name in os.listdir(f'{dataset_path}/{mode}/images'):
                    label_name = f'{img_name.split(".")[0]}.txt'
                    if os.path.isfile(f'{dataset_path}/{mode}/labels/{label_name}') == False:
                        print(f'이미지에 매칭되는 레이블이 없음: {mode}/{label_name}')
                        if fix:
                            # 빈 레이블 생성
                            with open(f'{dataset_path}/{mode}/labels/{label_name}', 'w') as f:
                                f.write('')
            return True
        except Exception as e:
            print(e)
            return False
                        
    def img_format(self, img_name):
        '''
        입력된 파일명이 이미지 포맷인지 검사하여 반환

        img_name: 입력된 파일의 이름
        '''
        if img_name.split('.')[-1] in ['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'tif', 'webp', 'ppm', 'pgm', 'pbm']:
            return True
        else:
            return False


    def label_format(self, label_name):
        '''
        입력된 파일명이 YOLO 레이블 포맷인지 검사하여 반환

        label_name: 입력된 레이블의 이름
        '''
        if label_name.split('.')[-1] == 'txt':
            return True
        else:
            return False