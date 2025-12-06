from src.IDCardRenderer import IDCardRenderer
from src.IDCardDataGenerator import IDCardDataGenerator
from src.IDCardAugmentor import IDCardAugmentor
import os
import json
import argparse
from pathlib import Path
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description='Generate Thai ID card OCR dataset')
    parser.add_argument('--output', type=str, default='outputs', help='Output directory (default: outputs)')
    parser.add_argument('--num-images', type=int, default=80, help='Number of base images (default: 80)')
    parser.add_argument('--num-aug', type=int, default=3, help='Augmentations per image (default: 3)')
    parser.add_argument('--lang', type=str, default='all', choices=['th', 'en', 'all'],
                        help='Language fields to extract: th, en, or all (default: all)')
    args = parser.parse_args()

    num_images = args.num_images
    num_augmentations = args.num_aug
    template_path = 'template/personal-card-template.jpg'

    th_fields = ['FullNameTH', 'BirthdayTH', 'Religion', 'Address', 'DateOfIssueTH', 'DateOfExpiryTH']
    en_fields = ['Identification_Number', 'NameEN', 'LastNameEN', 'BirthdayEN', 'DateOfIssueEN', 'DateOfExpiryEN']

    if args.lang == 'th':
        selected_fields = th_fields
    elif args.lang == 'en':
        selected_fields = en_fields
    else:
        selected_fields = th_fields + en_fields

    base_dir = f'{args.output}/base'
    augmented_dir = f'{args.output}/augmented_cards'
    final_dir = f'{args.output}/final_dataset'

    os.makedirs(f'{base_dir}/labels', exist_ok=True)
    os.makedirs(f'{augmented_dir}/images', exist_ok=True)
    os.makedirs(f'{augmented_dir}/labels_bbox', exist_ok=True)
    os.makedirs(f'{final_dir}/images', exist_ok=True)

    render_config = 'configs/identity_card/config-for-feature-extraction.json'
    label_config = 'configs/identity_card/config.json'
    template_path = 'template/personal-card-template.jpg'

    generator = IDCardDataGenerator(
        male_names_path='datasets/thai-names-corpus/male_names_th.txt',
        female_names_path='datasets/thai-names-corpus/female_names_th.txt',
        family_names_path='datasets/thai-names-corpus/family_names_th.txt',
        address_data_path='datasets/thai-province/province_with_district_and_sub_district.json'
    )

    renderer = IDCardRenderer(
        config_path=render_config,
        font_paths={
            'thai': ['fonts/dilleniaupc/DilleniaUPC Bold.ttf'],
            'english': ['fonts/dilleniaupc/DilleniaUPC Bold.ttf']
        }
    )

    if not renderer.load_image(template_path):
        print(f"Error: Cannot load template from {template_path}")
        return

    augmentor = IDCardAugmentor(
        num_augmentations_per_image=num_augmentations
    )

    with open(label_config, 'r', encoding='utf-8') as f:
        config = json.load(f)
    field_definitions = config['roi_extract']['front']

    total_cards = num_images * (1 + num_augmentations) if num_augmentations > 0 else num_images

    print("=" * 60)
    print("Setup completed")
    print(f"  Base images: {num_images}")
    print(f"  Augmentations per card: {num_augmentations}")
    print(f"  Total cards: {total_cards} (base + augmented)")
    print(f"  Selected fields: {len(selected_fields)} ({args.lang})")
    print(f"  Expected final images: {total_cards * len(selected_fields)}")
    print("=" * 60)

    print("\nGenerating base images...")
    generate_base_images(
        num_images=num_images,
        generator=generator,
        renderer=renderer,
        field_definitions=field_definitions,
        output_dir=base_dir,
        template_path=template_path
    )

    if num_augmentations > 0:
        print("\nAugmenting full cards...")
        augment_full_cards(
            base_dir=base_dir,
            augmentor=augmentor,
            output_dir=augmented_dir
        )
        source_dirs = [base_dir, augmented_dir]
        print(
            f"  Using base + augmented: {num_images} + {num_images * num_augmentations} = {num_images * (1 + num_augmentations)} images")
    else:
        print("\nSkipping augmentation (num-aug=0)")
        source_dirs = [base_dir]

    print("\nCropping fields to final dataset...")
    crop_fields_to_dataset(
        source_dirs=source_dirs,
        output_dir=final_dir,
        selected_fields=selected_fields
    )


def generate_base_images(num_images, generator, renderer, field_definitions,
                         output_dir, template_path):
    for i in tqdm(range(num_images), desc="Generating base images"):
        sample_data = generator.generate(
            gender='random',
            marital_status='random',
            age_range=(18, 85)
        )

        if not renderer.load_image(template_path):
            print(f"Error: Cannot reload template for image {i}")
            continue

        renderer.render_data(sample_data)

        image_name = f'card_{i:04d}.jpg'
        image_path = os.path.join(output_dir, image_name)
        renderer.save(image_path)

        boxes = []
        for idx, field in enumerate(field_definitions):
            field_name = field['name']
            text = sample_data.get(field_name, "")

            boxes.append({
                'class_id': idx,
                'class_name': field_name,
                'bbox': field['point'],
                'text': text
            })

        label_data = {'boxes': boxes}
        label_path = os.path.join(output_dir, 'labels', f'card_{i:04d}.json')
        with open(label_path, 'w', encoding='utf-8') as f:
            json.dump(label_data, f, ensure_ascii=False, indent=2)

    print(f"  Generated {num_images} base images")


def augment_full_cards(base_dir, augmentor, output_dir):
    base_image_files = list(Path(base_dir).glob('*.jpg'))
    base_image_files.sort()

    augmentor.process_files(
        base_image_files,
        output_dir
    )

    augmented_images = list(Path(f'{output_dir}/images').glob('*.jpg'))
    print(f"  Generated {len(augmented_images)} augmented images")


def crop_fields_to_dataset(source_dirs, output_dir, selected_fields):
    import cv2

    all_images = []
    for source_dir in source_dirs:
        images_dir = f'{source_dir}/images' if source_dir.endswith('augmented_cards') else source_dir
        labels_dir = f'{source_dir}/labels_bbox' if source_dir.endswith('augmented_cards') else f'{source_dir}/labels'

        for img_path in sorted(Path(images_dir).glob('*.jpg')):
            label_path = Path(labels_dir) / f'{img_path.stem}.json'
            if label_path.exists():
                all_images.append((img_path, label_path))

    field_counter = 0
    labels_data = []

    for img_path, label_path in tqdm(all_images, desc="Cropping fields"):
        image = cv2.imread(str(img_path))
        if image is None:
            continue

        with open(label_path, 'r', encoding='utf-8') as f:
            label_data = json.load(f)

        for box in label_data['boxes']:
            class_name = box.get('class_name', '')

            if class_name not in selected_fields:
                continue

            bbox = box['bbox']
            text = box.get('text', '')

            x1, y1, x2, y2 = map(int, bbox)

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(image.shape[1], x2)
            y2 = min(image.shape[0], y2)

            if x2 <= x1 or y2 <= y1:
                continue

            field_img = image[y1:y2, x1:x2]

            field_filename = f'field_{field_counter:05d}.jpg'
            field_path = os.path.join(output_dir, 'images', field_filename)
            cv2.imwrite(field_path, field_img)

            labels_data.append(f'{field_filename} {text}')
            field_counter += 1

    labels_txt_path = os.path.join(output_dir, 'labels.txt')
    with open(labels_txt_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(labels_data))

    print(f"  Cropped {field_counter} field images")
    print(f"  Labels saved to: {labels_txt_path}")


if __name__ == "__main__":
    main()