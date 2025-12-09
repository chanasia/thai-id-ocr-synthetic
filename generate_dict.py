import argparse
import sys
import os
from src.IDCardDataGenerator import IDCardDataGenerator
from tqdm import tqdm


def filter_fields(data, lang):
    th_fields = ['FullNameTH', 'BirthdayTH', 'Religion', 'Address', 'DateOfIssueTH', 'DateOfExpiryTH']
    en_fields = ['Identification_Number', 'NameEN', 'LastNameEN', 'BirthdayEN', 'DateOfIssueEN', 'DateOfExpiryEN']

    if lang == 'th':
        return {k: data[k] for k in th_fields if k in data}
    elif lang == 'en':
        return {k: data[k] for k in en_fields if k in data}
    else:
        all_fields = th_fields + en_fields
        return {k: data[k] for k in all_fields if k in data}


def generate_dict_file(count, lang, output_file):
    generator = IDCardDataGenerator(
        male_names_path='datasets/thai-names-corpus/male_names_th.txt',
        female_names_path='datasets/thai-names-corpus/female_names_th.txt',
        family_names_path='datasets/thai-names-corpus/family_names_th.txt',
        address_data_path='datasets/thai-province/province_with_district_and_sub_district.json',
        streets_data_path='datasets/thai-province/thai_streets_all.json'
    )

    texts = set()

    for _ in tqdm(range(count), desc="Generating data", unit="card"):
        data = generator.generate()
        filtered = filter_fields(data, lang)

        for value in filtered.values():
            if isinstance(value, str):
                texts.add(value.strip())

    with open(output_file, 'w', encoding='utf-8') as f:
        for text in sorted(texts):
            f.write(f"{text}\n")

    print(f"Generated {len(texts)} unique entries to {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Generate dictionary from IDCardDataGenerator')
    parser.add_argument('-c', '--count', type=int, default=1000,
                        help='Number of ID cards to generate (default: 1000)')
    parser.add_argument('-l', '--lang', choices=['th', 'en', 'all'], default='all',
                        help='Language selection: th, en, or all (default: all)')
    parser.add_argument('-o', '--output', type=str, default='dict_output.txt',
                        help='Output file path (default: dict_output.txt)')

    args = parser.parse_args()

    generate_dict_file(args.count, args.lang, args.output)


if __name__ == '__main__':
    main()