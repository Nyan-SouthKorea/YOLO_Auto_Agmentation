from setuptools import setup, find_packages

# README 파일을 UTF-8 인코딩으로 읽어옵니다.
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='yolo_auto_agmentation',
    version='1.1.3',
    description='enter dataset path with train, val, test. Then automatically augment every images to ready YOLO object detection train',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Na-I-Eon',
    author_email='112fkdldjs@naver.com',
    url='https://github.com/Nyan-SouthKorea/yolo_auto_augmentation',
    packages=find_packages(),
    install_requires=[
        'opencv-python', 'tqdm', 'albumentations', 'natsort'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
)
