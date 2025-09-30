# OCR and Post-OCR Pipeline for Historical Map Data Processing

A comprehensive pipeline for processing historical map images through OCR and post-OCR correction using low-resource approaches with LoRA fine-tuned models.

## Overview

This pipeline processes images from an input folder and outputs structured results to an output folder. Each image gets its own folder containing:

- OCR results (CSV with bboxes and text)
- Bbox clusters (CSV with clustered bounding boxes)
- Post-OCR results (JSON with LLM-corrected text and merged bboxes)
- Visualization images

## Features

- **OCR Processing**: Uses PaddleOCR with dynamic tiling for large images
- **Smart Clustering**: Groups overlapping OCR detections into clusters, keeps individual results separate
- **Post-OCR Correction**: Uses LoRA fine-tuned models for:
  - Bbox merging (using bboxes LoRA model) - for overlapping detections
  - Text correction (using labels LoRA model) - for text improvements
- **Hybrid Processing**: Processes clusters with LLM, keeps individual results unchanged
- **Structured Output**: Each image gets its own output folder with organized results

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure you have the LoRA models in the correct location:

LoRa models can get from this notebooks version 8 (labels) and 10 (bboxes): [Kaggle Notebook](https://www.kaggle.com/code/hngnguynhuy/grpo-map-name)

```
src/post_ocr_module/grpo_model/
├── bboxes/
│   ├── adapter_config.json
│   └── adapter_model.safetensors
└── labels/
    ├── adapter_config.json
    └── adapter_model.safetensors
```

## Usage

### Basic Usage

1. Place your images in the `input/` folder
2. Run the pipeline:
```bash
python main.py
```

### Advanced Usage

```bash
python pipeline.py --input /path/to/images --output /path/to/output
```

### Command Line Options

- `--input, -i`: Input directory containing images (default: input)
- `--output, -o`: Output directory for results (default: output)

## Output Structure

```
output/
├── image1/
│   ├── ocr_results.csv          # Raw OCR results
│   ├── bbox_clusters.csv        # Clustered bounding boxes
│   ├── post_ocr_results.json    # LLM-processed results
│   ├── final_results.csv        # Final corrected results
│   └── visualization.png        # Visual overlay
├── image2/
│   └── ...
└── processing_summary.json       # Overall processing summary
```

## Pipeline Components

### 1. OCR Module (`src/ocr_module/ocr_model.py`)
- Uses PaddleOCR for text detection and recognition
- Implements dynamic tiling for large images
- Includes NMS (Non-Maximum Suppression) for duplicate removal

### 2. Post-OCR Module (`src/post_ocr_module/infer.py`)
- Uses LoRA fine-tuned models for post-processing
- Bbox clustering and merging
- Text correction and improvement
- Supports both bbox and label processing tasks

### 3. Main Pipeline (`pipeline.py`)
- Orchestrates the entire process
- Handles file I/O and directory structure
- Creates visualizations and summaries

## Configuration

### OCR Parameters
- `min_score`: Minimum confidence score for OCR detections (default: 0.2)
- `iou_merge`: IoU threshold for merging overlapping detections (default: 0.2)

### LLM Parameters
- `model_name`: Base model for LoRA fine-tuning (default: "unsloth/Qwen2.5-3B-Instruct")
- `max_seq_length`: Maximum sequence length (default: 4096)
- `lora_rank`: LoRA rank for fine-tuning (default: 64)

## Example Output

### OCR Results CSV
```csv
bbox,text,score
"[[100, 50], [200, 50], [200, 80], [100, 80]]","Hồ Chí Minh",0.95
```

### Bbox Clusters CSV
```csv
id,type,bbox,text,score,bbox_count,texts
cluster_0,cluster,"[[95, 45], [205, 45], [205, 85], [95, 85]]","Hồ; Chí Minh",0.95,2,"[""Hồ"", ""Chí Minh""]"
individual_0,individual,"[[300, 100], [400, 100], [400, 130], [300, 130]]","Sài Gòn",0.88,1,"[""Sài Gòn""]"
```

### Post-OCR Results JSON
```json
{
  "processed_clusters": [
    {
      "cluster_id": 0,
      "cluster_type": "bbox",
      "original_bboxes": [...],
      "merged_bbox": "(406, 57, 566, 592)",
      "texts": ["Hồ", "Chí Minh"]
    }
  ],
  "total_clusters": 1,
  "bbox_clusters": 1,
  "label_clusters": 0
}
```

### Final Results CSV
```csv
id,type,original_bbox,corrected_bbox,text,corrected_text,score,correction_type,bbox_count
cluster_0,cluster,"[[95, 45], [205, 45], [205, 85], [95, 85]]","(406, 57, 566, 592)","Hồ; Chí Minh","Hồ; Chí Minh",0.95,"bbox_merge",2
individual_0,individual,"[[300, 100], [400, 100], [400, 130], [300, 130]]","[[300, 100], [400, 100], [400, 130], [300, 130]]","Sài Gòn","Sài Gòn",0.88,"none",1
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce `gpu_memory_utilization` in the PostOCRProcessor
2. **LoRA Model Not Found**: Ensure LoRA models are in the correct directory structure
3. **No Images Found**: Check that images are in the input directory with supported formats (.jpg, .png, etc.)

### Supported Image Formats
- JPEG (.jpg, .jpeg)
- PNG (.png)
- BMP (.bmp)
- TIFF (.tiff, .tif)

## Development

### Project Structure
```
├── pipeline.py                 # Main pipeline script
├── main.py                     # Entry point
├── requirements.txt           # Dependencies
├── input/                     # Input images
├── output/                    # Output results
└── src/
    ├── ocr_module/
    │   └── ocr_model.py       # OCR processing
    └── post_ocr_module/
        ├── infer.py           # Post-OCR processing
        └── grpo_model/        # LoRA models
            ├── bboxes/
            └── labels/
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Citation

If you use this pipeline in your research, please cite:

```bibtex
@software{ocr_post_ocr_pipeline,
  title={OCR and Post-OCR Pipeline for Historical Map Data Processing},
  author={Nguyen Huy Hung},
  year={2025},
  url={https://github.com/whistle-hikhi/Correcting-Post-OCR-with-low-resources-approach-in-Historical-Map-data-in-Vietnam}
}
```