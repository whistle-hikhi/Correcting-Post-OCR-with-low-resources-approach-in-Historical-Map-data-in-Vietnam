#!/usr/bin/env python3
"""
Main entry point for OCR and Post-OCR Pipeline
"""

import sys
import os
from pipeline import OCRPostOCRPipeline

def main():
    """Main function to run the OCR and Post-OCR pipeline"""
    # Default paths
    input_dir = "input"
    output_dir = "output"
    
    # Check if input directory exists and has images
    if not os.path.exists(input_dir):
        print(f"Error: Input directory '{input_dir}' does not exist!")
        print("Please create the input directory and add your images.")
        sys.exit(1)
    
    # Create pipeline and run
    print("Starting OCR and Post-OCR Pipeline...")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    
    try:
        pipeline = OCRPostOCRPipeline(input_dir, output_dir)
        pipeline.run_pipeline()
        print("Pipeline completed successfully!")
    except Exception as e:
        print(f"Error running pipeline: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
