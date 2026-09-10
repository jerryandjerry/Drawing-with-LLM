#| export

import re
import vtracer
from IPython.display import SVG, display, Image
from PIL import Image as PILImage

def convert_paths_to_polygons(svg_code: str, size: int) -> str:
    # Find all path elements
    scale_factor = 384 / size

    path_pattern = re.compile(
        r'<path[^>]*d="([^"]+)"[^>]*fill="([^"]+)"[^>]*transform="translate\(([^)]+)\)"[^>]*/?>'
    )
    paths = path_pattern.findall(svg_code)

    polygons = []

    for d_content, fill_color, translate in paths:
        tx, ty = map(float, translate.split(','))
        points = []
        tokens = re.findall(r'[MLZmlz]|-?\d+\.?\d*,\-?\d+\.?\d*', d_content.strip())
        for token in tokens:
            if token in {'M', 'L', 'Z', 'm', 'l', 'z'}:
                continue
            x_str, y_str = token.split(',')
            x = int(round(float(x_str) + tx))
            y = int(round(float(y_str) + ty))
            points.append(f"{x},{y}")

        if points:
            polygon = f'<polygon points="{" ".join(points)}" fill="{fill_color}"/>'
            polygons.append(polygon)

    # Build the compact SVG
    new_svg = (
        f'<svg width="384" height="384" viewBox="0 0 384 384"><g transform="scale({scale_factor})">'
        + "".join(polygons)
        + '</g></svg>'
    )

    return new_svg


def fix_svg_size(svg_code: str, target_size: int = 384) -> str:
    """
    Update the <svg> tag's width and height to match target size.
    """
    # Replace width attribute
    svg_code = re.sub(
        r'(width\s*=\s*")([^"]+)(")',
        lambda m: f'{m.group(1)}{target_size}{m.group(3)}',
        svg_code,
        count=1
    )
    # Replace height attribute
    svg_code = re.sub(
        r'(height\s*=\s*")([^"]+)(")',
        lambda m: f'{m.group(1)}{target_size}{m.group(3)}',
        svg_code,
        count=1
    )
    return svg_code


def bitmap_to_svg_layered(img, input_path, output_path, resolution):
    default_svg = """<svg width="384" height="384" viewBox="0 0 384 384"><g transform="scale(3.0)"><polygon points="0,0 128,0 128,128 0,128" fill="#58208E"/><polygon points="85,0 128,0 128,128 0,128 0,111 2,111 2,113 4,113 4,115 7,113 11,114 14,115 16,114 16,112 19,112 21,116 21,114 24,114 25,110 35,111 49,114 57,117 56,113 62,114 63,107 71,102 82,99 100,99 100,97 96,97 95,29 96,27 89,27 87,25 88,22 91,20 95,20 95,15 87,17 80,17 78,16 79,14 80,11 84,11 84,9 89,9 89,4 82,4 82,2" fill="#4A1A6C"/><polygon points="0,0 40,0 40,2 45,2 44,8 34,7 34,4 33,7 26,6 26,8 31,9 30,12 27,11 27,15 24,15 24,22 29,21 28,13 37,11 38,10 44,10 44,12 50,12 47,16 37,15 36,17 32,16 31,28 34,29 32,30 31,93 28,92 29,23 24,25 23,31 23,44 27,40 25,45 23,47 22,91 28,93 31,95 32,99 26,100 24,100 19,101 19,97 17,99 14,97 14,95 12,95 11,99 9,99 6,96 7,101 5,103 4,96 2,99 2,95 0,95" fill="#221146"/><polygon points="40,0 85,0 86,2 82,2 82,4 89,4 89,9 84,9 84,11 80,11 80,14 82,15 78,16 87,16 89,15 95,15 95,20 88,22 88,27 84,27 84,29 80,28 80,33 78,33 77,37 75,37 75,39 85,37 90,34 94,33 95,37 93,38 95,38 95,49 86,52 85,48 83,53 79,55 67,54 58,51 51,48 45,45 48,43 45,42 46,40 44,34 41,40 43,41 40,41 40,43 42,44 36,43 34,39 31,40 31,29 33,29 31,28 31,15 36,16 37,15 47,15 49,13 44,12 44,10 39,11 37,12 32,13 29,14 29,21 24,22 24,15 26,13 27,11 30,11 31,9 24,9 26,8 26,6 33,7 33,5 31,4 34,4 34,7 44,8 44,3 40,2" fill="#F296B2"/><polygon points="117,0 128,0 128,95 121,94 119,78 118,66 117,38" fill="#371C72"/><polygon points="40,0 85,0 86,2 82,2 82,4 89,4 89,9 84,9 84,11 80,11 80,14 82,15 78,16 87,16 89,15 95,15 95,20 83,20 58,21 58,19 61,18 73,17 77,16 76,15 71,15 71,13 45,18 43,19 31,19 32,15 36,16 37,15 47,15 49,13 44,12 44,10 39,11 37,12 32,13 29,14 29,21 24,22 24,15 26,13 27,11 30,11 31,9 24,9 26,8 26,6 33,7 33,5 31,4 34,4 34,7 44,8 44,3 40,2" fill="#963AE1"/><polygon points="11,0 20,0 19,5 17,5 17,7 12,8 13,10 16,11 16,13 10,14 10,16 14,17 14,15 18,14 19,17 19,49 18,83 15,93 11,93 9,90 8,85 8,44 8,42 9,23 9,9 11,6 9,5" fill="#522289"/><polygon points="66,13 71,13 71,15 77,14 77,16 73,18 58,19 58,21 83,19 94,19 90,22 88,22 87,25 87,23 71,26 71,29 68,31 60,32 60,34 55,36 45,36 44,34 41,40 43,41 40,41 40,43 42,44 36,43 34,39 31,40 31,29 33,29 31,28 31,19 43,18 59,14" fill="#D667CE"/><polygon points="77,104 85,104 84,106 88,108 84,109 84,114 91,115 91,113 99,112 99,116 104,113 106,113 106,115 110,114 117,115 119,114 120,116 123,116 123,114 128,113 128,128 81,128 84,125 84,121 74,117 70,113 64,111 65,107 72,105" fill="#200E3F"/><polygon points="51,48 60,51 73,54 79,54 84,52 84,65 81,65 81,67 84,68 82,70 82,75 76,77 75,83 74,80 72,80 71,83 69,86 69,80 61,81 58,78 58,71 56,71 56,76 52,74 53,66 52,62 51,52" fill="#7648C8"/><polygon points="112,0 117,0 119,66 121,94 127,94 128,99 126,100 126,98 123,101 112,101 104,100 106,96 111,93 114,93 113,76 112,72 110,71 112,66 112,36 109,34 110,32 113,34 112,15 106,12 107,22 106,41 105,41 105,7 109,11 112,12" fill="#170F3B"/><polygon points="25,110 35,111 48,114 47,119 50,121 47,122 42,119 42,121 36,119 37,126 39,125 38,123 43,123 46,128 0,128 0,111 2,111 2,113 4,113 4,115 7,113 11,114 14,115 16,114 16,112 19,112 21,116 21,114 24,114" fill="#240B4B"/><polygon points="43,30 45,34 47,41 46,42 48,43 47,45 51,46 52,58 53,64 51,67 52,69 50,69 51,74 50,75 45,75 46,73 44,72 44,77 50,77 54,82 53,85 49,86 49,89 45,89 43,85 40,82 36,80 35,78 32,77 32,63 35,62 36,63 36,60 36,54 39,51 39,48 42,46 40,44 41,37" fill="#34146F"/><polygon points="106,12 112,14 113,15 113,34 110,34 113,36 113,55 111,55 110,58 108,58 108,50 106,50 106,80 105,80 104,54 103,60 100,60 99,65 98,65 97,47 97,25 100,25 99,21 97,20 97,15 104,13 105,27 106,21" fill="#653B94"/><polygon points="88,26 96,27 96,61 95,72 93,72 93,75 91,74 90,71 90,64 87,65 86,60 84,47 86,48 86,52 91,49 95,49 95,38 92,40 88,39 92,37 95,37 94,34 87,36 84,38 75,39 75,37 77,37 78,33 80,33 80,28 84,29 84,27" fill="#7749B8"/><polygon points="11,0 20,0 19,5 17,5 17,7 12,8 13,10 16,11 16,13 10,14 10,16 14,17 14,15 18,14 19,17 19,39 16,51 13,51 12,47 13,44 15,44 15,42 13,42 13,36 9,37 9,9 11,6 9,5" fill="#B855DC"/><polygon points="41,86 47,89 54,92 55,96 63,97 69,99 70,100 95,96 100,97 100,99 82,100 71,103 64,107 62,114 57,114 57,117 50,116 44,107 46,106 54,105 57,103 64,102 48,96 48,94 44,92 44,90 37,89 36,87 41,88" fill="#A92DA5"/><polygon points="51,48 60,51 78,55 75,57 66,60 75,61 73,63 69,64 79,65 80,72 75,73 57,63 53,62 51,56" fill="#9962DB"/><polygon points="0,0 6,0 6,35 5,67 4,67 3,49 2,67 0,67" fill="#3D218A"/><polygon points="117,11 119,11 119,13 125,14 125,30 126,37 126,55 124,51 123,45 122,47 120,47 120,51 118,51 117,38" fill="#5F2F8B"/><polygon points="106,50 108,50 108,58 112,55 113,56 113,66 111,71 112,72 113,82 108,82 109,93 105,95 105,80" fill="#43298E"/><polygon points="102,13 104,13 104,32 102,32 102,35 100,35 100,37 102,37 102,46 97,47 97,25 100,25 99,21 97,20 97,15" fill="#9B3EB2"/><polygon points="60,104 65,104 62,114 57,114 57,117 50,116 44,107 46,106 54,105" fill="#9B3E66"/><polygon points="0,0 6,0 6,35 5,67 4,67 2,6 0,6" fill="#3A216F"/><polygon points="25,23 29,23 29,54 26,52 26,41 22,45 22,29 24,24" fill="#6E3B96"/><polygon points="97,47 98,47 99,63 100,60 103,60 103,84 98,86 97,72" fill="#3F2387"/><polygon points="32,39 34,39 38,43 43,45 41,49 39,50 39,53 38,56 36,56 37,59 34,59 33,61 31,56 31,40" fill="#754BC3"/><polygon points="12,27 14,27 14,31 16,31 17,29 19,33 18,44 16,51 13,51 12,47 13,44 15,44 15,42 13,42 13,36 9,37 9,30" fill="#6C44BC"/><polygon points="93,5 95,6 94,7 99,9 99,11 97,11 97,13 88,14 87,17 80,17 78,16 79,14 80,11 91,9 91,6" fill="#260D50"/><polygon points="11,0 20,0 19,5 17,5 17,7 12,8 13,10 16,11 16,13 10,14 10,16 12,16 11,20 12,22 10,24 9,23 9,9 11,6 9,5" fill="#8633D9"/><polygon points="47,24 65,24 62,26 50,28 45,28 45,34 43,30 42,29 33,29 31,27 38,25" fill="#A749CE"/><polygon points="28,100 36,102 41,104 42,107 41,109 25,106 21,102" fill="#C422A9"/><polygon points="107,23 113,26 113,34 110,34 112,35 111,39 109,38 109,40 112,41 111,44 111,42 107,44" fill="#E993B5"/><polygon points="21,70 22,70 22,91 28,93 31,95 32,99 26,100 23,100 21,95" fill="#3F1B54"/><polygon points="80,28 89,28 91,29 89,33 87,32 81,39 75,39 75,37 77,37 78,33 80,33" fill="#45215E"/><polygon points="2,5 3,5 3,32 0,34 0,6" fill="#9D42DB"/><polygon points="106,50 107,50 108,76 109,93 105,95 105,80" fill="#181141"/><polygon points="20,0 21,0 21,11 20,14 14,15 14,17 10,16 10,14 16,13 16,11 12,10 12,8 17,7 17,5 19,5" fill="#2E1266"/><polygon points="103,27 104,27 104,32 102,32 102,35 100,35 100,37 102,37 102,46 98,47 98,29" fill="#EE8EB6"/><polygon points="37,101 45,101 52,103 57,103 58,105 50,107 43,106 36,103" fill="#541A68"/><polygon points="106,0 112,0 112,12 107,11 106,10" fill="#832ECE"/><polygon points="34,96 45,97 53,100 57,101 57,103 50,104 35,98" fill="#C22CA7"/><polygon points="88,26 96,27 96,33 86,37 83,37 87,31 90,31 91,29 84,28" fill="#9C5A8D"/><polygon points="25,23 29,23 29,37 24,37 24,31 25,29 24,29 24,24" fill="#E270D2"/><polygon points="2,68 4,68 3,77 2,80 5,81 4,86 0,84 0,69" fill="#601A8F"/><polygon points="85,0 100,0 99,3 94,4 82,4 82,2" fill="#220B4F"/><polygon points="121,25 125,25 124,30 123,35 118,35 118,26" fill="#AF5EAB"/><polygon points="118,11 119,13 125,14 124,18 122,20 123,22 121,24 120,23 118,25" fill="#DF60CF"/><polygon points="106,12 112,14 112,23 106,21" fill="#F077CB"/><polygon points="60,104 65,104 65,106 62,106 62,108 53,111 50,110 54,105" fill="#CF5F74"/><polygon points="103,8 104,12 102,12 101,15 96,15 96,20 95,20 95,15 88,16 88,14 94,13 97,13 97,11 101,11 101,9" fill="#581F84"/><polygon points="0,0 6,0 5,16 4,16 2,6 0,6" fill="#431A82"/><polygon points="100,60 103,60 103,72 101,71 101,69 99,69 98,71 98,65" fill="#5637AD"/><polygon points="18,15 20,15 20,37 22,37 23,33 23,40 19,39" fill="#3D1F5D"/><polygon points="13,24 19,24 19,33 16,31 14,31" fill="#E276CF"/><polygon points="118,0 124,0 125,5 124,6 118,6" fill="#A63DF0"/><polygon points="66,52 83,52 79,55 67,54" fill="#EF95BE"/><polygon points="107,23 113,26 113,34 111,33 112,30 107,29" fill="#815483"/><polygon points="26,37 29,37 29,54 26,52 27,41 28,39 26,39" fill="#7C4FC9"/><polygon points="86,28 91,29 89,33 86,33 81,33 81,30" fill="#240E46"/><polygon points="102,24 104,25 102,29 98,29 98,47 97,47 97,28 101,28" fill="#B87296"/><polygon points="110,4 112,4 112,12 107,11 108,6" fill="#AC40E7"/><polygon points="48,47 50,47 50,54 49,56 45,56 45,52 47,50 46,48" fill="#7435B2"/><polygon points="118,36 121,36 123,37 122,41 120,41 120,43 122,44 119,44 119,49 118,49" fill="#6F39A1"/><polygon points="0,22 3,25 3,32 0,34" fill="#633DB3"/><polygon points="102,18 104,18 104,25 102,25 101,27 97,27 97,25 100,24" fill="#E676C3"/><polygon points="19,8 20,10 17,13 19,14 14,15 14,17 10,16 10,14 16,13 15,10" fill="#4E1E8E"/><polygon points="106,12 112,14 111,17 107,17 106,21" fill="#B143C5"/><polygon points="100,1 104,2 104,6 99,7 97,4 100,4" fill="#7F33C6"/><polygon points="23,81 27,82 29,84 29,88 26,87 26,84 23,83" fill="#9125AE"/><polygon points="11,0 16,0 14,4 9,6" fill="#6624C5"/><polygon points="24,16 26,17 29,18 29,21 24,22" fill="#B94FE7"/><polygon points="107,23 112,25 111,29 107,27" fill="#F293C0"/><polygon points="97,15 103,16 100,19 97,19" fill="#BC49D4"/><polygon points="103,8 104,12 98,13 99,11 101,11 101,9" fill="#A23CD4"/><polygon points="23,25 25,28 27,27 25,31 22,31" fill="#5D2C67"/><polygon points="102,24 104,25 102,29 97,29 101,28" fill="#92508C"/><polygon points="121,1 125,3 125,5 121,4 119,2" fill="#8530CC"/><polygon points="24,45 26,46 26,49 24,49 22,51 23,46" fill="#46206E"/><polygon points="33,14 38,14 36,17 32,16" fill="#63229A"/><polygon points="0,0 3,0 2,4 0,4" fill="#7B2DE1"/><polygon points="13,97 15,98 14,102 12,101" fill="#331253"/></g><path d="M20 364 L24 356 L28 364 M22 360 L26 360" stroke="#CCCCCC"/><path d="M364 28 L360 20 L356 28 M362 24 L358 24" stroke="#333333"/></svg>"""

    # Step 1: Resize the input image to 256x256
    # 256x256 is helpful bc each length od path is shorter, so with the same max_svg_length, we can have more paths(more color)
    size = resolution
    img = img.resize((size,size), PILImage.LANCZOS)
    img.save(input_path)

    # Step 2: Convert the resized image to SVG
    max_svg_length = 9996  # target length limit

    # Define injection to prevent OCR hallucination
    injection_a1 = '<path d="M20 364 L24 356 L28 364 M22 360 L26 360" stroke="#CCCCCC"/>'      # Bottom-left A
    injection_a2 = '<path d="M364 28 L360 20 L356 28 M362 24 L358 24" stroke="#888888"/>'         # Top-right A
    injection = injection_a1 + injection_a2
    injection_length = len(injection)

    # Binary search for best layer_difference
    low = 1
    high = 200
    best_svg_code = None
    best_layer_difference = 10

    try:
        while low <= high:
            layer_difference = (low + high) // 2
    
            vtracer.convert_image_to_svg_py(
                input_path,
                output_path,
                colormode='color',        # Options: 'color' or 'binary'
                hierarchical='stacked',   # Options: 'stacked' or 'cutout'
                mode='polygon',           # Options: 'spline', 'polygon', or 'none'
                filter_speckle=3,          # remove tiny regions
                color_precision=8,         # reduce color complexity
                layer_difference=layer_difference,  # more aggressive merging
                corner_threshold=10,       # remove subtle corners
                length_threshold=10,       # remove short paths
                max_iterations=10,         # faster, less detail
                splice_threshold=10,       # simplify curves
                path_precision=3           # reduce vertex detail
            )
    
            # Step 3: Read and display the SVG
            try:
                with open(output_path, "r", encoding="utf-8") as f:
                    svg_code = f.read()
            except UnicodeDecodeError:
                # Bad SVG output (probably corrupted), try again
                high = layer_difference - 1
                continue  # go back to binary search
    
            # Clean the first two lines if present
            lines = svg_code.splitlines()
            removed_length = 0
            if lines and lines[0].strip().startswith('<?xml'):
                removed_length += len(lines[0]) + 1  # +1 for newline
                lines = lines[1:]
            if lines and lines[0].strip().startswith('<!--'):
                removed_length += len(lines[0]) + 1  # +1 for newline
                lines = lines[1:]
            svg_code = "\n".join(lines)
    
            svg_code = svg_code.replace(
                '<svg ',
                '<svg viewBox="0 0 384 384" ', #this is 20 bytes more
                1  # only replace first occurrence
            )
            
            # remove version and xmlns attributes
            svg_code = re.sub(r'\s*version="[^"]*"', '', svg_code)
            svg_code = re.sub(r'\s*xmlns="[^"]*"', '', svg_code)
    
            # use polygon instead of path
            svg_code = convert_paths_to_polygons(svg_code, size)
                
            # Correct length check: give credit for removed lines
            if len(svg_code) + injection_length <= max_svg_length:
                best_svg_code = svg_code
                best_layer_difference = layer_difference
                high = layer_difference - 1  # search for even more detail
            else:
                low = layer_difference + 1  # simplify more


        # ============= if layer_diff = 200 is not enough, keep increasing ==========================================
        if best_svg_code == None:
            layer_difference = 210
            max_diff = 1000 
            while layer_difference <= max_diff:
                vtracer.convert_image_to_svg_py(
                    input_path,
                    output_path,
                    colormode='color',        # Options: 'color' or 'binary'
                    hierarchical='stacked',   # Options: 'stacked' or 'cutout'
                    mode='polygon',           # Options: 'spline', 'polygon', or 'none'
                    filter_speckle=3,          # remove tiny regions
                    color_precision=8,         # reduce color complexity
                    layer_difference=layer_difference,  # more aggressive merging
                    corner_threshold=10,       # remove subtle corners
                    length_threshold=10,       # remove short paths
                    max_iterations=10,         # faster, less detail
                    splice_threshold=10,       # simplify curves
                    path_precision=3           # reduce vertex detail
                )
        
                try:
                    with open(output_path, "r", encoding="utf-8") as f:
                        svg_code = f.read()
                except UnicodeDecodeError:
                    # Bad SVG output (probably corrupted), try again
                    continue  # go back to start of while loop
        
                # Clean the first two lines if present
                lines = svg_code.splitlines()
                removed_length = 0
                if lines and lines[0].strip().startswith('<?xml'):
                    removed_length += len(lines[0]) + 1  # +1 for newline
                    lines = lines[1:]
                if lines and lines[0].strip().startswith('<!--'):
                    removed_length += len(lines[0]) + 1  # +1 for newline
                    lines = lines[1:]
                svg_code = "\n".join(lines)
        
                svg_code = svg_code.replace(
                    '<svg ',
                    '<svg viewBox="0 0 384 384" ', #this is 20 bytes more
                    1  # only replace first occurrence
                )
                
                # remove version and xmlns attributes
                svg_code = re.sub(r'\s*version="[^"]*"', '', svg_code)
                svg_code = re.sub(r'\s*xmlns="[^"]*"', '', svg_code)
        
                # use polygon instead of path
                svg_code = convert_paths_to_polygons(svg_code, size)
                print(f'{len(svg_code)=}')
                    
                # Correct length check: give credit for removed lines
                if len(svg_code) + injection_length <= max_svg_length: # acount for injection
                    best_svg_code = svg_code
                    best_layer_difference = layer_difference
                    break
                else:
                    layer_difference += 10
    
        svg_code = fix_svg_size(best_svg_code, target_size=384) # doesnt change length
        # Inject fake letter path to prevent OCR hallucination
        svg_code = svg_code.replace("</svg>", injection + "</svg>")
        print(f'{best_layer_difference=}')

    except Exception as e:
        print(f"{e}")
        svg_code = default_svg

    
    return svg_code

# ================== color richfullness =========================
import os
from PIL import Image
import numpy as np
from skimage.color import rgb2lab
from scipy.stats import entropy
import matplotlib.pyplot as plt
def compute_color_richness_entropy(image, bins=32):
    # Ensure image is RGB and resized
    image = image.convert('RGB')
    image = image.resize((256, 256))
    img_array = np.array(image) / 255.0
    lab_image = rgb2lab(img_array)

    # Flatten LAB channels
    L = lab_image[:, :, 0].flatten()
    A = lab_image[:, :, 1].flatten()
    B = lab_image[:, :, 2].flatten()

    # Compute histograms
    L_hist, _ = np.histogram(L, bins=bins, density=True)
    A_hist, _ = np.histogram(A, bins=bins, density=True)
    B_hist, _ = np.histogram(B, bins=bins, density=True)

    # Calculate entropy for each channel
    L_entropy = entropy(L_hist + 1e-8)
    A_entropy = entropy(A_hist + 1e-8)
    B_entropy = entropy(B_hist + 1e-8)

    total_entropy = L_entropy + A_entropy + B_entropy

    # Normalize: range from 0-10.39
    max_entropy = 10.39 # = 3 * np.log(bins)
    normalized_entropy = total_entropy / max_entropy
    
    return normalized_entropy

def map_score_to_range(normalized_score):
    if normalized_score <= 0.5:
        return 384
    elif normalized_score >= 0.7:
        return 128
    else:
        # Linear interpolation between 384 and 128
        alpha = (normalized_score - 0.5) / (0.7 - 0.5)
        return int(384 - alpha * (384 - 128))
