from yolo_auto_agment import Img_aug

img_aug = Img_aug()

img_aug.auto_augment(dataset_path='test_dataset', repeat=3)

img_aug.auto_draw(auged_path='test_dataset_aug')