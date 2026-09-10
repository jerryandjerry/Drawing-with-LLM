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
    default_svg = """<svg width="384" height="384" viewBox="0 0 384 384"><circle cx="50" cy="50" r="40" fill="red" /></svg>"""

    # Step 1: Resize the input image to 256x256
    # 256x256 is helpful bc each length od path is shorter, so with the same max_svg_length, we can have more paths(more color)
    size = resolution
    img = img.resize((size,size), PILImage.LANCZOS)
    img.save(input_path)

    # Step 2: Convert the resized image to SVG
    max_svg_length = 9996  # target length limit

    # Define injection to prevent OCR hallucination
    injection = ''
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
