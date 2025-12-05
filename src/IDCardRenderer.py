import cv2
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from . import constants

class IDCardRenderer:
    def __init__(self, config_path, font_paths=None):
        self.config = self._load_config(config_path)
        
        self.font_paths = font_paths or {
            'thai': ['fonts/dilleniaupc/DilleniaUPC Bold.ttf'],
            'english': ['fonts/dilleniaupc/DilleniaUPC Bold.ttf']
        }

        self.img = None
        self.img_pil = None
        self.draw = None

    def _load_config(self, config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
        
    def load_image(self, img_path):
        self.img = cv2.imread(img_path)
        if self.img is None:
            print(f"Error: Cannot load image from {img_path}")
            return False
        return True
    
    def _load_font(self, font_list, size):
        for font_path in font_list:
            try:
                return ImageFont.truetype(font_path, size)
            except Exception as e:
                continue
        print(f"Warning: Could not load any font, using default")
        return ImageFont.load_default()
    
    def _wrap_text(self, text, font, max_width):
        lines = []
        words = text.split(' ')
        current_line = []
        
        for word in words:
            test_line = ' '.join(current_line + [word])
            bbox = self.draw.textbbox((0, 0), test_line, font=font)
            test_width = bbox[2] - bbox[0]
            
            if test_width <= max_width:
                current_line.append(word)
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                current_line = [word]
        
        if current_line:
            lines.append(' '.join(current_line))
        
        return lines

    def _draw_multiline_text(self, position, text, font, color, box_width, 
                            box_height, line_spacing=1.2, first_line_indent=0):
        x, y = position
        lines = self._wrap_text(text, font, box_width - 6)
        
        bbox = self.draw.textbbox((0, 0), "A", font=font)
        line_height = (bbox[3] - bbox[1]) * line_spacing
        
        current_y = y
        
        for idx, line in enumerate(lines):
            indent = first_line_indent if idx == 0 else 0
            self.draw.text((x + indent, current_y), line, font=font, fill=color)
            current_y += line_height
        
        return len(lines)
    
    def _get_font_for_field(self, field_name, font_size):
        thai_fields = ["FullNameTH", "BirthdayTH", "Religion", "Address", 
                      "DateOfIssueTH", "DateOfExpiryTH"]
        
        if field_name in thai_fields:
            return self._load_font(self.font_paths['thai'], font_size)
        else:
            return self._load_font(self.font_paths['english'], font_size)
        
    def render_data(self, data):
        if self.img is None:
            print("Error: No image loaded. Call load_image() first")
            return
        
        img_with_data = self.img.copy()
        self.img_pil = Image.fromarray(cv2.cvtColor(img_with_data, cv2.COLOR_BGR2RGB))
        self.draw = ImageDraw.Draw(self.img_pil)
        
        front_fields = self.config['roi_extract']['front']
        
        for field in front_fields:
            field_name = field['name']
            x1, y1, x2, y2 = field['point']
            
            text = data.get(field_name, "TEST")
            box_width = x2 - x1
            box_height = y2 - y1
            
            font_size = constants.FONT_SIZES.get(field_name, 24)
            font = self._get_font_for_field(field_name, font_size)
            text_color = constants.FONT_COLORS.get(field_name, (0, 0, 0))
            
            if field_name == "Address":
                self._draw_multiline_text(
                    (x1 + 3, y1), 
                    text, 
                    font, 
                    text_color, 
                    box_width, 
                    box_height,
                    line_spacing=2.2,
                    first_line_indent=33
                )
            else:
                self.draw.text((x1 + 3, y1), text, font=font, fill=text_color)
        
        self.img_with_data = cv2.cvtColor(np.array(self.img_pil), cv2.COLOR_RGB2BGR)

    def show(self, title='ID Card with Sample Data'):
        if not hasattr(self, 'img_with_data'):
            print("Error: No rendered data. Call render_data() first")
            return
        
        plt.figure(figsize=(15, 8))
        plt.imshow(cv2.cvtColor(self.img_with_data, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.title(title)
        plt.tight_layout()
        plt.show()

    def save(self, output_path):
        if not hasattr(self, 'img_with_data'):
            print("Error: No rendered data. Call render_data() first")
            return False
        
        cv2.imwrite(output_path, self.img_with_data)
        return True