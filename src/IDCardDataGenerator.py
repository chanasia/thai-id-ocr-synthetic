import random
from pythainlp.transliterate import romanize
from typing import Dict, List
from . import constants
from datetime import datetime, timedelta
import json
from dateutil.relativedelta import relativedelta

class IDCardDataGenerator:
    def __init__(self, 
            male_names_path='../datasets/thai-names-corpus/male_names_th.txt',
            female_names_path='../datasets/thai-names-corpus/female_names_th.txt',
            family_names_path='../datasets/thai-names-corpus/family_names_th.txt',
            address_data_path='../datasets/thai-province/province_with_district_and_sub_district.json',
            streets_data_path='../datasets/thai-province/thai_streets_all.json'):
        self.male_names = self._load_names(male_names_path)
        self.female_names = self._load_names(female_names_path)
        self.family_names = self._load_names(family_names_path)
        self.current_date = datetime.now()
        self.address_data = self._load_address_data(address_data_path)
        self.streets_data = self._load_streets_data(streets_data_path)

    def _load_address_data(self, filepath: str):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load address data: {e}")
            return []

    def _load_names(self, filepath: str) -> List[str]:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                names = [line.strip() for line in f if line.strip()]
            return names
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return []

    def _load_streets_data(self, filepath: str):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load streets data: {e}")
            return {}

    def _transliterate_name(self, thai_name: str) -> str:
        try:
            english = romanize(thai_name, engine='thai2rom')
            return english.capitalize()
        except:
            pass
        
        try:
            english = romanize(thai_name, engine='thai2rom_onnx')
            return english.capitalize()
        except:
            pass
        
        try:
            english = romanize(thai_name, engine='royin')
            return english.capitalize()
        except Exception as e:
            # print(f"Warning: Could not transliterate '{thai_name}': {e}")
            return thai_name
        
    def generate_name(self, gender: str = 'random', 
        marital_status: str = 'random') -> Dict[str, str]:

        if gender == 'random':
            gender = random.choice(['male', 'female'])

        if gender == 'male':
            first_name = random.choice(self.male_names)
            title_prefix = constants.TITLE_PREFIXES['male']
        else:
            first_name = random.choice(self.female_names)

            if marital_status == 'random':
                marital_status = random.choice(['single', 'married'])
            
            if marital_status == 'married':
                title_prefix = constants.TITLE_PREFIXES['female_married']
            else:
                title_prefix = constants.TITLE_PREFIXES['female_single']

        last_name = random.choice(self.family_names)

        full_name_th = f"{title_prefix['th']} {first_name} {last_name}"

        first_name_en = self._transliterate_name(first_name)
        last_name_en = self._transliterate_name(last_name)

        first_name_en = first_name_en.capitalize()
        last_name_en = last_name_en.capitalize()

        return {
            'FullNameTH': full_name_th,
            'NameEN': f"{title_prefix['en']} {first_name_en}",
            'LastNameEN': last_name_en,
            '_first_name_th': first_name,
            '_last_name_th': last_name,
            '_first_name_en': first_name_en,
            '_last_name_en': last_name_en,
            '_gender': gender,
            '_title_prefix': title_prefix
        }

    def generate_multiple_names(self, count: int = 10) -> List[Dict[str, str]]:
        names = []
        for _ in range(count):
            name = self.generate_name()
            names.append(name)
        return names
    
    def print_name_example(self, name_dict: Dict[str, str]):
        print(f"{name_dict['FullNameTH']}, {name_dict['NameEN']} {name_dict['LastNameEN']}")

    def _random_date_between(self, start_date: datetime, end_date: datetime) -> datetime:
        time_between = end_date - start_date
        days_between = time_between.days
        random_days = random.randint(0, days_between)
        return start_date + timedelta(days=random_days)
    
    def _format_thai_date(self, date: datetime, use_short_month: bool = True) -> str:
        day = date.day
        month_dict = constants.THAI_MONTHS if use_short_month else constants.THAI_MONTHS_FULL
        month = month_dict[date.month]
        year = date.year + 543  # แปลง ค.ศ. เป็น พ.ศ.
        
        return f"{day} {month} {year}"
    
    def _format_english_date(self, date: datetime) -> str:
        day = date.day
        month = constants.ENGLISH_MONTHS[date.month]
        year = date.year
        
        return f"{day} {month} {year}"
    
    def generate_dates(self, 
                  age_range: tuple = (18, 85),
                  issue_years_ago_range: tuple = (0, 10),
                  card_validity_years: int = 10) -> dict:
        
        min_age, max_age = age_range
        
        max_birth_date = self.current_date - timedelta(days=min_age * 365.25)
        min_birth_date = self.current_date - timedelta(days=max_age * 365.25)
        
        birth_date = self._random_date_between(min_birth_date, max_birth_date)
        
        earliest_issue = birth_date + timedelta(days=18 * 365.25)
        
        min_issue_ago, max_issue_ago = issue_years_ago_range
        latest_issue = self.current_date - timedelta(days=min_issue_ago * 365.25)
        earliest_possible_issue = self.current_date - timedelta(days=max_issue_ago * 365.25)
        
        actual_earliest_issue = max(earliest_issue, earliest_possible_issue)
        
        if actual_earliest_issue > latest_issue:
            issue_date = latest_issue
        else:
            issue_date = self._random_date_between(actual_earliest_issue, latest_issue)

        expiry_date = issue_date + relativedelta(years=card_validity_years)
        
        age = (self.current_date - birth_date).days // 365
        
        return {
            'BirthdayTH': self._format_thai_date(birth_date),
            'BirthdayEN': self._format_english_date(birth_date),
            'DateOfIssueTH': self._format_thai_date(issue_date),
            'DateOfIssueEN': self._format_english_date(issue_date),
            'DateOfExpiryTH': self._format_thai_date(expiry_date),
            'DateOfExpiryEN': self._format_english_date(expiry_date),
            '_birth_date': birth_date,
            '_issue_date': issue_date,
            '_expiry_date': expiry_date,
            '_age': age
        }

    @staticmethod
    def generate_thai_id(formatted=False):
        digits = [random.randint(1, 9)]
        digits.extend([random.randint(0, 9) for _ in range(11)])
        
        total = sum(d * (13 - i) for i, d in enumerate(digits))
        checksum = (11 - (total % 11)) % 10
        
        digits.append(checksum)
        id_number = ''.join(map(str, digits))
        
        if formatted:
            return f"{id_number[0]} {id_number[1:5]} {id_number[5:10]} {id_number[10:12]} {id_number[12]}"
        
        return id_number
    
    @staticmethod
    def validate_thai_id(id_number):
        id_clean = id_number.replace(' ', '').replace('-', '')
        
        if len(id_clean) != 13 or not id_clean.isdigit():
            return False
        
        total = sum(int(id_clean[i]) * (13 - i) for i in range(12))
        expected_checksum = (11 - (total % 11)) % 10
        
        return int(id_clean[12]) == expected_checksum
    
    def generate_religion(self) -> str:
        religions = [
            ('พุทธ', 94.0),
            ('อิสลาม', 5.0),
            ('คริสต์', 0.7),
            ('ฮินดู', 0.2),
            ('ซิกข์', 0.1)
        ]
        
        religion_names = [r[0] for r in religions]
        religion_weights = [r[1] for r in religions]
        
        return random.choices(religion_names, weights=religion_weights)[0]

    def generate_address(self) -> dict:
        if not self.address_data:
            return {'Address': 'บ้านเลขที่ 123 ถนนสุขุมวิท แขวงคลองเตย เขตคลองเตย กรุงเทพมหานคร'}

        province = random.choice(self.address_data)
        province_name_th = province['name_th']

        if not province.get('districts'):
            district_name_th = ""
            sub_district_name_th = ""
        else:
            district = random.choice(province['districts'])
            district_name_th = district['name_th']

            if not district.get('sub_districts'):
                sub_district_name_th = ""
            else:
                sub_district = random.choice(district['sub_districts'])
                sub_district_name_th = sub_district['name_th']

        house_number = self._generate_house_number()

        street = ""
        if self.streets_data and province_name_th in self.streets_data:
            streets_list = self.streets_data[province_name_th].get('all_streets', [])
            if streets_list:
                selected_street = random.choice(streets_list)
                selected_street = selected_street.replace('ซอย', 'ซ.')
                selected_street = selected_street.replace('ถนน', 'ถ.')
                street = f" {selected_street}"

        is_bangkok = province_name_th == "กรุงเทพมหานคร"

        if is_bangkok:
            district_prefix = "" if district_name_th.startswith("เขต") else "เขต"
            address = f"{house_number}{street} แขวง{sub_district_name_th} {district_prefix}{district_name_th} {province_name_th}"
        else:
            tambon = f" ต.{sub_district_name_th}" if sub_district_name_th else ""
            amphoe = f" อ.{district_name_th}" if district_name_th else ""
            address = f"{house_number}{street}{tambon}{amphoe} จ.{province_name_th}"

        return {
            'Address': address.strip(),
            '_province': province_name_th,
            '_district': district_name_th,
            '_sub_district': sub_district_name_th,
            '_house_number': house_number
        }

    def _generate_house_number(self) -> str:
        # รูปแบบต่างๆ
        formats = [
            lambda: str(random.randint(1, 999)),  # 123
            lambda: f"{random.randint(1, 999)}/{random.randint(1, 99)}",  # 123/45
            lambda: f"{random.randint(1, 99)}/{random.randint(1, 9)}",  # 12/3
        ]
        
        return random.choice(formats)()
    
    def generate(self, 
        gender: str = 'random',
        marital_status: str = 'random',
        age_range: tuple = (18, 85)) -> dict:
        
        name_data = self.generate_name(gender, marital_status)
        
        date_data = self.generate_dates(age_range=age_range)

        id_number = self.generate_thai_id(formatted=True)

        religion = self.generate_religion()

        address_data = self.generate_address()
        
        return {
            **name_data,
            **date_data,
            **address_data,
            'Identification_Number': id_number,
            '_id_number_raw': id_number.replace(' ', ''),
            'Religion': religion,
        }