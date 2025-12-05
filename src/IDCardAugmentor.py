import cv2
import json
import numpy as np
import albumentations as A
from pathlib import Path
import os
from tqdm import tqdm
import random


class IDCardAugmentor:
    def __init__(self,
                 image_size=(600, 350),
                 num_augmentations_per_image=10):

        self.image_size = image_size
        self.num_augmentations_per_image = num_augmentations_per_image
        self.transform = self._create_transform()

    def _create_transform(self):
        bg_color = [random.randint(200, 255) for _ in range(3)]

        return A.Compose([
            A.Resize(
                height=int(self.image_size[1] * 0.97),
                width=int(self.image_size[0] * 0.97),
                p=1.0
            ),

            A.PadIfNeeded(
                min_height=self.image_size[1],
                min_width=self.image_size[0],
                border_mode=cv2.BORDER_CONSTANT,
                fill=bg_color,
                p=1.0,
            ),

            A.Rotate(limit=1, p=0.5),
            A.Perspective(
                scale=(0.01, 0.02),
                fit_output=False,
                keep_size=True,
                p=0.3,
            ),

            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.5
            ),
            A.RGBShift(
                r_shift_limit=15,
                g_shift_limit=15,
                b_shift_limit=15,
                p=0.3
            ),

            A.GaussNoise(
                std_range=[0.1, 0.2],
                mean_range=[0, 0],
                per_channel=True,
                noise_scale_factor=1,
                p=0.5
            )

        ], bbox_params=A.BboxParams(
            format='pascal_voc',
            min_visibility=0.3,
            min_area=50,
            label_fields=['class_ids']
        ))

    def _validate_bbox(self, bbox, img_width, img_height, min_size=5):
        x1, y1, x2, y2 = bbox

        width = x2 - x1
        height = y2 - y1

        if width < min_size or height < min_size:
            return False, "Too small"

        aspect_ratio = width / max(height, 1)
        if aspect_ratio > 30 or aspect_ratio < 0.03:
            return False, f"Bad aspect ratio: {aspect_ratio:.2f}"

        if x1 < 0 or y1 < 0 or x2 > img_width or y2 > img_height:
            return False, "Out of bounds"

        return True, "OK"

    def augment_image(self, image, bboxes, class_ids, class_names, texts):
        augmented_images = []
        augmented_bboxes_list = []
        augmented_class_names_list = []
        augmented_texts_list = []

        attempts = 0
        max_attempts = self.num_augmentations_per_image * 3

        while len(augmented_images) < self.num_augmentations_per_image and attempts < max_attempts:
            attempts += 1

            try:
                transformed = self.transform(
                    image=image,
                    bboxes=bboxes,
                    class_ids=class_ids
                )

                h, w = transformed['image'].shape[:2]
                all_valid = True

                if len(transformed['bboxes']) != len(bboxes):
                    continue

                for bbox in transformed['bboxes']:
                    is_valid, msg = self._validate_bbox(bbox, w, h)
                    if not is_valid:
                        all_valid = False
                        break

                if all_valid:
                    augmented_images.append(transformed['image'])
                    augmented_bboxes_list.append(transformed['bboxes'])
                    augmented_class_names_list.append(class_names)
                    augmented_texts_list.append(texts)

            except Exception as e:
                continue

        return augmented_images, augmented_bboxes_list, augmented_class_names_list, augmented_texts_list

    def _save_augmented_data(self, image, bboxes, class_ids, class_names, texts, output_name, output_images_dir,
                             output_labels_dir):

        h, w = image.shape[:2]
        if (w, h) != self.image_size:
            scale_x = self.image_size[0] / w
            scale_y = self.image_size[1] / h

            image = cv2.resize(image, self.image_size)

            scaled_bboxes = []
            for bbox in bboxes:
                x1, y1, x2, y2 = bbox
                scaled_bbox = [
                    x1 * scale_x,
                    y1 * scale_y,
                    x2 * scale_x,
                    y2 * scale_y
                ]
                scaled_bboxes.append(scaled_bbox)
            bboxes = scaled_bboxes

        img_path = f'{output_images_dir}/{output_name}.jpg'
        cv2.imwrite(img_path, image)

        label_path = f'{output_labels_dir}/{output_name}.json'
        label_data = {
            'image_size': {'width': image.shape[1], 'height': image.shape[0]},
            'boxes': []
        }

        for bbox, class_id, class_name, text in zip(bboxes, class_ids, class_names, texts):
            label_data['boxes'].append({
                'class_id': int(class_id),
                'class_name': class_name,
                'bbox': [float(x) for x in bbox],
                'text': text
            })

        with open(label_path, 'w', encoding='utf-8') as f:
            json.dump(label_data, f, ensure_ascii=False, indent=2)

    def process_files(self, image_files, output_dir):
        output_images_dir = f'{output_dir}/images'
        output_labels_dir = f'{output_dir}/labels_bbox'

        os.makedirs(output_images_dir, exist_ok=True)
        os.makedirs(output_labels_dir, exist_ok=True)

        for img_path in tqdm(image_files, desc="Augmenting"):
            image = cv2.imread(str(img_path))

            if image is None:
                continue

            if image.shape[:2][::-1] != self.image_size:
                image = cv2.resize(image, self.image_size)

            label_path = img_path.parent / 'labels' / f'{img_path.stem}.json'

            if not label_path.exists():
                print(f"Warning: Label not found for {img_path.name}, skipping")
                continue

            with open(label_path, 'r', encoding='utf-8') as f:
                label_data = json.load(f)

            bboxes = [box['bbox'] for box in label_data['boxes']]
            class_ids = [box['class_id'] for box in label_data['boxes']]
            class_names = [box['class_name'] for box in label_data['boxes']]
            texts = [box.get('text', '') for box in label_data['boxes']]

            aug_images, aug_bboxes_list, aug_class_names_list, aug_texts_list = self.augment_image(
                image,
                bboxes,
                class_ids,
                class_names,
                texts
            )

            base_name = img_path.stem
            for aug_idx, (aug_img, aug_bboxes, aug_class_names, aug_texts) in enumerate(
                    zip(aug_images, aug_bboxes_list, aug_class_names_list, aug_texts_list)
            ):
                output_name = f'{base_name}_aug_{aug_idx:03d}'
                self._save_augmented_data(
                    aug_img,
                    aug_bboxes,
                    class_ids,
                    aug_class_names,
                    aug_texts,
                    output_name,
                    output_images_dir,
                    output_labels_dir
                )

    def process_dataset(self, input_images_dir, output_dir):
        output_images_dir = f'{output_dir}/images'
        output_labels_dir = f'{output_dir}/labels_bbox'

        os.makedirs(output_images_dir, exist_ok=True)
        os.makedirs(output_labels_dir, exist_ok=True)

        image_files = list(Path(input_images_dir).glob('*.jpg'))
        image_files += list(Path(input_images_dir).glob('*.png'))

        if len(image_files) == 0:
            print(f"Error: No images found in {input_images_dir}")
            return

        total_generated = 0

        for img_idx, img_path in enumerate(tqdm(image_files, desc="Processing images")):
            image = cv2.imread(str(img_path))

            if image is None:
                continue

            if image.shape[:2][::-1] != self.image_size:
                image = cv2.resize(image, self.image_size)

            label_path = Path(input_images_dir) / 'labels' / f'{img_path.stem}.json'

            if not label_path.exists():
                print(f"Warning: Label not found for {img_path.name}, skipping")
                continue

            with open(label_path, 'r', encoding='utf-8') as f:
                label_data = json.load(f)

            bboxes = [box['bbox'] for box in label_data['boxes']]
            class_ids = [box['class_id'] for box in label_data['boxes']]
            class_names = [box['class_name'] for box in label_data['boxes']]
            texts = [box.get('text', '') for box in label_data['boxes']]

            aug_images, aug_bboxes_list, aug_class_names_list, aug_texts_list = self.augment_image(
                image,
                bboxes,
                class_ids,
                class_names,
                texts
            )

            base_name = img_path.stem
            for aug_idx, (aug_img, aug_bboxes, aug_class_names, aug_texts) in enumerate(
                    zip(aug_images, aug_bboxes_list, aug_class_names_list, aug_texts_list)
            ):
                output_name = f'{base_name}_aug_{aug_idx:03d}'
                self._save_augmented_data(
                    aug_img,
                    aug_bboxes,
                    class_ids,
                    aug_class_names,
                    aug_texts,
                    output_name,
                    output_images_dir,
                    output_labels_dir
                )
                total_generated += 1