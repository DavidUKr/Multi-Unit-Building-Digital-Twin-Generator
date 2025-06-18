from PIL import Image
import io
from io import BytesIO
import base64
import numpy as np
import cv2

def tensor_to_pngs(spatial, target_width=0, target_height=0, threshold=0.5):
    """
    Convert a spatial tensor (HxWxC) to binary mask PNGs (base64-encoded).
    Each channel is treated as an independent binary mask.
    Args:
        spatial: Torch tensor of shape (H, W, C), e.g., (1060, 1130, C), containing mask values (e.g., 0-1 or probabilities).
        target_width: Int, desired width for resizing (0 means no resizing).
        target_height: Int, desired height for resizing (0 means no resizing).
        threshold: Float, threshold for binarizing the channel (default: 0.5). Ignored if input is already binary.
    Returns:
        List of base64-encoded PNG strings, one binary mask per channel (0 and 255).
    """
    # Ensure tensor is on CPU and converted to NumPy
    spatial = spatial.cpu().numpy()  # Shape: (1060, 1130, C)
    
    # Initialize list for base64-encoded PNGs
    pngs = []
    
    # Process each channel
    for c in range(spatial.shape[0]):
        # Extract channel
        channel = spatial[c, :, :]  # shape (W,H)
        
        # Binarize channel (if not already binary)
        if not np.all(np.logical_or(channel == 0, channel == 1)):  # Check if non-binary
            channel = (channel >= threshold).astype(np.uint8)  # Threshold to 0s and 1s
        else:
            channel = channel.astype(np.uint8)  # Ensure uint8 type
        
        # Upsample binary mask if target dimensions are specified
        if target_width > 0 and target_height > 0:
            channel = cv2.resize(channel, (target_width, target_height), interpolation=cv2.INTER_NEAREST)
        
        # Scale to 0 and 255 for PNG
        channel = channel * 255
        
        # Create grayscale PNG
        img = Image.fromarray(channel, mode="L")  # Grayscale image
        png_base64 = png_to_base64(img)
        pngs.append(png_base64)
    
    return pngs

def png_to_base64(png):
    buffer = io.BytesIO()
    png.save(buffer, format="PNG")
    png_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return png_base64

def base64_to_Image(base64_string):
    img_data = base64.b64decode(base64_string)
    return Image.open(BytesIO(img_data))

def resize_tensor(input, target_width, target_height):
    return cv2.resize(input, (target_width, target_height), interpolation=cv2.INTER_AREA)

def split(input, num_horizontal_splits=10, num_vertical_splits=8):
    """
    Slices an image into smaller parts and returns a nested dictionary of base64 strings.
    The second-to-last tile in a row/column absorbs extra pixels if the image dimensions
    are not perfectly divisible by the number of splits.

    Args:
        input (PIL.Image): Input image.
        num_horizontal_splits (int): Number of splits along the width.
        num_vertical_splits (int): Number of splits along the height.

    Returns:
        dict: Nested dictionary like {'row_i': {'column_j': base64, ...}, ...}
    """
    width, height = input.size

    if num_horizontal_splits < 1 or num_vertical_splits < 1:
        raise ValueError("Number of splits must be at least 1 in each dimension.")
    if width < num_horizontal_splits:
        raise ValueError(f"Image width ({width}px) is less than the number of horizontal splits ({num_horizontal_splits}).")
    if height < num_vertical_splits:
        raise ValueError(f"Image height ({height}px) is less than the number of vertical splits ({num_vertical_splits}).")

    base_tile_width = width // num_horizontal_splits
    base_tile_height = height // num_vertical_splits
    remainder_width = width % num_horizontal_splits
    remainder_height = height % num_vertical_splits

    # print(f"Image dimensions: {width}x{height}")
    # print(f"Target splits: {num_horizontal_splits}x{num_vertical_splits}")
    # print(f"Base tile size: {base_tile_width}x{base_tile_height}")
    if remainder_width > 0 or remainder_height > 0:
        print(f"Remainder (absorbed by second-to-last tile): width={remainder_width}, height={remainder_height}")

    output = {}
    tile_count = 0
    current_y = 0
    for i in range(num_vertical_splits):
        output[f'row_{i}'] = {}  # Initialize row dictionary
        current_x = 0
        current_row_height = base_tile_height
        if remainder_height > 0 and i == num_vertical_splits - 2:
            current_row_height += remainder_height

        next_y = min(current_y + current_row_height, height)
        effective_row_height = next_y - current_y

        if effective_row_height <= 0:
            continue

        for j in range(num_horizontal_splits):
            current_col_width = base_tile_width
            if remainder_width > 0 and j == num_horizontal_splits - 2:
                current_col_width += remainder_width

            next_x = min(current_x + current_col_width, width)
            effective_col_width = next_x - current_x

            if effective_col_width <= 0:
                current_x = next_x
                continue

            bbox = (current_x, current_y, next_x, next_y)
            try:
                tile = input.crop(bbox)
                output[f'row_{i}'][f'column_{j}'] = png_to_base64(tile)
                tile_count += 1
            except Exception as e:
                print(f"Error processing tile ({i},{j}) with box {bbox}: {e}")
                raise

            current_x = next_x

        current_y = next_y

    if not output:
        raise ValueError("No tiles generated. Check image dimensions and splits.")

    return output

def file_to_image(image_file, tranform_to_rgb=False, to_PIL_Image=False):
    image_data = np.frombuffer(image_file.read(), np.uint8)
    image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
    if tranform_to_rgb:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    if to_PIL_Image:
        image = Image.fromarray(image)

    return image

def reconstruct(tiles, full_width, full_height, tiles_width, tiles_height):
    # Calculate tile sizes (accounting for uneven divisions)
    base_tile_width = full_width // tiles_width  # Base width for most tiles
    base_tile_height = full_height // tiles_height  # Base height for most tiles

    # Store tile sizes for each position
    tile_widths = [base_tile_width] * tiles_width
    tile_heights = [base_tile_height] * tiles_height

    # Adjust the last column and row to account for remaining pixels
    tile_widths[-1] = full_width - (tiles_width - 1) * base_tile_width
    tile_heights[-1] = full_height - (tiles_height - 1) * base_tile_height

    # Create a new blank image for reconstruction
    reconstructed_image = Image.new("RGB", (full_width, full_height))

    for y in range(tiles_height):
        for x in range(tiles_width):
            tile=tiles[f'row_{y}'][f'column_{x}'] #TODO: check the structure of tiles

            # Calculate the paste position
            paste_x = x * base_tile_width
            paste_y = y * base_tile_height
            
            # Paste the tile onto the reconstructed image
            reconstructed_image.paste(tile, (paste_x, paste_y))

    return reconstructed_image