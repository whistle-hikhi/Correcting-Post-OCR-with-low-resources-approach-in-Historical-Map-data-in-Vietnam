#!/usr/bin/env python3
"""
Example usage of the OCR and Post-OCR Pipeline
"""

import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from pipeline import OCRPostOCRPipeline

def create_sample_structure():
    """Create sample directory structure for demonstration"""
    # Create input directory if it doesn't exist
    input_dir = Path("input")
    input_dir.mkdir(exist_ok=True)
    
    # Create output directory if it doesn't exist
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    print("📁 Directory structure created:")
    print(f"   Input: {input_dir.absolute()}")
    print(f"   Output: {output_dir.absolute()}")
    print("\n💡 To use the pipeline:")
    print("   1. Add your images to the 'input' folder")
    print("   2. Run: python main.py")
    print("   3. Check results in the 'output' folder")

def demonstrate_pipeline():
    """Demonstrate the pipeline with sample data"""
    print("🚀 OCR and Post-OCR Pipeline Demo")
    print("=" * 50)
    
    # Check if input directory has images
    input_dir = Path("input")
    if not input_dir.exists():
        print("❌ Input directory not found. Creating it...")
        input_dir.mkdir(exist_ok=True)
    
    # Get image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    image_files = []
    
    for file_path in input_dir.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in image_extensions:
            image_files.append(file_path)
    
    if not image_files:
        print("📷 No images found in input directory.")
        print("   Please add some images (.jpg, .png, etc.) to the 'input' folder")
        print("   Then run: python main.py")
        return
    
    print(f"📷 Found {len(image_files)} images to process:")
    for img in image_files:
        print(f"   - {img.name}")
    
    # Create pipeline
    try:
        pipeline = OCRPostOCRPipeline("input", "output")
        print("\n🔧 Pipeline initialized successfully!")
        print("   Ready to process images...")
        
        # Note: We don't actually run the pipeline here to avoid dependencies
        print("\n⚠️  Note: This is a demonstration.")
        print("   To actually run the pipeline, ensure all dependencies are installed:")
        print("   pip install -r requirements.txt")
        print("   Then run: python main.py")
        
    except Exception as e:
        print(f"❌ Error initializing pipeline: {str(e)}")
        print("   Make sure all dependencies are installed: pip install -r requirements.txt")

def show_output_structure():
    """Show the expected output structure"""
    print("\n📂 Expected Output Structure:")
    print("output/")
    print("├── image1/")
    print("│   ├── ocr_results.csv          # Raw OCR results")
    print("│   ├── bbox_clusters.csv        # Clustered bounding boxes")
    print("│   ├── post_ocr_results.json    # LLM-processed results")
    print("│   └── visualization.png        # Visual overlay")
    print("├── image2/")
    print("│   └── ...")
    print("└── processing_summary.json       # Overall processing summary")

def main():
    """Main function"""
    print("🔍 OCR and Post-OCR Pipeline - Example Usage")
    print("=" * 60)
    
    # Create sample structure
    create_sample_structure()
    
    # Demonstrate pipeline
    demonstrate_pipeline()
    
    # Show output structure
    show_output_structure()
    
    print("\n✅ Setup complete!")
    print("   Next steps:")
    print("   1. Add images to the 'input' folder")
    print("   2. Install dependencies: pip install -r requirements.txt")
    print("   3. Run the pipeline: python main.py")

if __name__ == "__main__":
    main()
