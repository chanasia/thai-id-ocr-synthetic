# Thai ID Card OCR Synthetic Dataset Generator

Synthetic dataset generator for Thai ID card OCR training

## Features

- Generate synthetic Thai national ID card images
- Support both Thai and English fields
- Customizable augmentation pipeline
- Ready for OCR training (image + labels.txt format)

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic Command

```bash
python generate_dataset.py --output <output_dir> --num-images <count> --num-aug <count> --lang <language>
```

### Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--output` | str | `outputs` | Output directory |
| `--num-images` | int | `80` | Number of base images to generate |
| `--num-aug` | int | `3` | Augmentations per base image |
| `--lang` | str | `all` | Language fields: `th`, `en`, or `all` |

### Language Fields

**Thai (`--lang th`):** 6 fields
- FullNameTH
- BirthdayTH
- Religion
- Address
- DateOfIssueTH
- DateOfExpiryTH

**English (`--lang en`):** 6 fields
- Identification_Number
- NameEN
- LastNameEN
- BirthdayEN
- DateOfIssueEN
- DateOfExpiryEN

**All (`--lang all`):** 12 fields (Thai + English)

## Examples

### Generate 1,000 Thai field images (no augmentation)

```bash
python generate_dataset.py --output dataset_th --num-images 1000 --num-aug 0 --lang th
```

**Output:** 6,000 images (1,000 cards × 6 fields)

### Generate English fields only (with augmentation)

```bash
python generate_dataset.py --output dataset_en --num-images 1000 --num-aug 2 --lang en
```

**Output:** 24,000 images (2,000 cards × 12 fields)

## Output Structure

```
<output_dir>/
├── base/
│   ├── card_0000.jpg
│   ├── card_0001.jpg
│   └── labels/
│       ├── card_0000.json
│       └── card_0001.json
├── augmented_cards/
│   ├── images/
│   │   ├── card_0000_aug_000.jpg
│   │   └── card_0000_aug_001.jpg
│   └── labels_bbox/
│       └── card_0000_aug_000.json
└── final_dataset/           # ← Ready for EasyOCR
    ├── images/
    │   ├── field_00000.jpg
    │   ├── field_00001.jpg
    │   └── ...
    └── labels.txt
```

### labels.txt Format

```
field_00000.jpg นายสมชาย ใจดี
field_00001.jpg 1 2345 67890 12 3
field_00002.jpg Mr. Somchai
field_00003.jpg 15 Jan. 1990
```

## Pipeline

1. **Generate Base Images** - Create synthetic ID cards with random data
2. **Augmentation** - Apply rotation, perspective, noise, brightness
3. **Crop Fields** - Extract individual fields using bounding boxes
4. **Create Labels** - Generate  labels.txt