import os
from PIL import Image

def optimize_image(filepath, max_width=300):
    try:
        img = Image.open(filepath)
        original_size = os.path.getsize(filepath)
        
        # Resize if width is larger than max_width
        w, h = img.size
        if w > max_width:
            ratio = max_width / float(w)
            new_h = int(h * ratio)
            img = img.resize((max_width, new_h), Image.Resampling.LANCZOS)
        
        # Save as PNG with optimization and palettization (converting to P mode reduces size significantly)
        if img.mode != 'P':
            img = img.convert('RGBA')
            # Adaptive palette quantization
            img = img.quantize(colors=256).convert('RGBA')
            
        img.save(filepath, 'PNG', optimize=True)
        new_size = os.path.getsize(filepath)
        reduction = (original_size - new_size) / original_size * 100
        print(f"Optimized {os.path.basename(filepath)}: {original_size/1024:.1f}KB -> {new_size/1024:.1f}KB ({reduction:.1f}% reduction)")
    except Exception as e:
        print(f"Error optimizing {filepath}: {e}")

def main():
    crop_dir = r"static/crop_images"
    print("Optimizing crop images...")
    for filename in os.listdir(crop_dir):
        if filename.endswith(".png"):
            filepath = os.path.join(crop_dir, filename)
            optimize_image(filepath, max_width=300)
            
    # Also optimize main home logo and farming images
    print("\nOptimizing main static images...")
    optimize_image("static/farming.png", max_width=600)
    optimize_image("static/agrismart_logo.png", max_width=250)

if __name__ == "__main__":
    main()
