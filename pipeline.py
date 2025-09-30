#!/usr/bin/env python3
"""
OCR and Post-OCR Pipeline for Historical Map Data Processing
Processes images from input folder and outputs results to output folder
"""

import os
import sys
import json
import csv
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Any
import argparse
from PIL import Image
import logging

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from ocr_module.ocr_model import OCRProcessor
from post_ocr_module.infer import PostOCRProcessor

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OCRPostOCRPipeline:
    """Main pipeline class for OCR and post-OCR processing"""
    
    def __init__(self, input_dir: str, output_dir: str):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.ocr_processor = OCRProcessor()
        self.post_ocr_processor = PostOCRProcessor()
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def get_image_files(self) -> List[Path]:
        """Get all image files from input directory"""
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        image_files = []
        
        for file_path in self.input_dir.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in image_extensions:
                image_files.append(file_path)
                
        return sorted(image_files)
    
    def create_output_structure(self, image_name: str) -> Dict[str, Path]:
        """Create output folder structure for a single image"""
        image_output_dir = self.output_dir / image_name
        image_output_dir.mkdir(exist_ok=True)
        
        return {
            'base_dir': image_output_dir,
            'ocr_csv': image_output_dir / 'ocr_results.csv',
            'clusters_csv': image_output_dir / 'bbox_clusters.csv',
            'post_ocr_results': image_output_dir / 'post_ocr_results.json',
            'visualization': image_output_dir / 'visualization.png'
        }
    
    def process_single_image(self, image_path: Path) -> Dict[str, Any]:
        """Process a single image through OCR and post-OCR pipeline"""
        logger.info(f"Processing image: {image_path.name}")
        
        # Create output structure
        image_name = image_path.stem
        output_paths = self.create_output_structure(image_name)
        
        try:
            # Step 1: OCR Processing
            logger.info(f"Running OCR on {image_path.name}")
            ocr_results = self.ocr_processor.process_image(str(image_path))
            
            # Save OCR results to CSV
            self.save_ocr_results(ocr_results, output_paths['ocr_csv'])
            
            # Step 2: Bbox Clustering
            logger.info(f"Clustering bboxes for {image_path.name}")
            clusters, individual_results = self.cluster_bboxes(ocr_results)
            
            # Save clusters to CSV
            self.save_clusters(clusters, individual_results, output_paths['clusters_csv'])
            
            # Step 3: Post-OCR Processing with LLM (only for clusters)
            logger.info(f"Running post-OCR processing for {image_path.name}")
            post_ocr_results = None
            if clusters:
                post_ocr_results = self.post_ocr_processor.process_clusters(
                    clusters, 
                    ocr_results,
                    str(output_paths['post_ocr_results'])
                )
            else:
                # No clusters to process
                post_ocr_results = {
                    'processed_clusters': [],
                    'total_clusters': 0,
                    'bbox_clusters': 0,
                    'label_clusters': 0
                }
            
            # Step 4: Create visualization
            self.create_visualization(image_path, ocr_results, clusters, output_paths['visualization'])
            
            # Step 5: Create final corrected CSV
            self.create_final_csv(ocr_results, clusters, individual_results, post_ocr_results, output_paths['base_dir'] / 'final_results.csv')
            
            logger.info(f"Successfully processed {image_path.name}")
            return {
                'image_name': image_name,
                'status': 'success',
                'ocr_count': len(ocr_results),
                'cluster_count': len(clusters),
                'output_paths': {k: str(v) for k, v in output_paths.items()}
            }
            
        except Exception as e:
            logger.error(f"Error processing {image_path.name}: {str(e)}")
            return {
                'image_name': image_name,
                'status': 'error',
                'error': str(e)
            }
    
    def save_ocr_results(self, ocr_results: List[Dict], csv_path: Path):
        """Save OCR results to CSV file"""
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            if ocr_results:
                writer = csv.DictWriter(f, fieldnames=['bbox', 'text', 'score'])
                writer.writeheader()
                for result in ocr_results:
                    writer.writerow({
                        'bbox': json.dumps(result['bbox']),
                        'text': result['text'],
                        'score': result['score']
                    })
    
    def cluster_bboxes(self, ocr_results: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """Cluster bboxes based on spatial proximity and return both clusters and individual results"""
        if not ocr_results:
            return [], []
        
        # Convert bboxes to center points and areas for clustering
        bbox_data = []
        for i, result in enumerate(ocr_results):
            bbox = result['bbox']
            # Calculate center point and area
            x_coords = [point[0] for point in bbox]
            y_coords = [point[1] for point in bbox]
            center_x = sum(x_coords) / len(x_coords)
            center_y = sum(y_coords) / len(y_coords)
            area = self.calculate_bbox_area(bbox)
            
            bbox_data.append({
                'index': i,
                'center': (center_x, center_y),
                'area': area,
                'text': result['text'],
                'bbox': bbox,
                'score': result['score']
            })
        
        # Clustering based on spatial proximity and IoU
        clusters = []
        individual_results = []
        used_indices = set()
        
        for i, data in enumerate(bbox_data):
            if i in used_indices:
                continue
                
            cluster = [data]
            used_indices.add(i)
            
            # Find overlapping bboxes using IoU
            for j, other_data in enumerate(bbox_data):
                if j in used_indices or j == i:
                    continue
                
                # Calculate IoU between bboxes
                iou = self.calculate_bbox_iou(data['bbox'], other_data['bbox'])
                distance = self.calculate_distance(data['center'], other_data['center'])
                
                # Cluster if IoU > 0.1 or distance < 50 pixels
                if iou > 0.1 or distance < 50:
                    cluster.append(other_data)
                    used_indices.add(j)
            
            if len(cluster) > 1:
                # Multiple bboxes - create cluster
                clusters.append({
                    'cluster_id': len(clusters),
                    'bboxes': cluster,
                    'cluster_center': self.calculate_cluster_center(cluster),
                    'cluster_bbox': self.calculate_cluster_bbox(cluster),
                    'is_cluster': True
                })
            else:
                # Single bbox - keep as individual result
                individual_results.append({
                    'index': data['index'],
                    'bbox': data['bbox'],
                    'text': data['text'],
                    'score': data['score'],
                    'is_cluster': False
                })
        
        return clusters, individual_results
    
    def calculate_bbox_area(self, bbox: List[List[float]]) -> float:
        """Calculate area of a bbox"""
        x_coords = [point[0] for point in bbox]
        y_coords = [point[1] for point in bbox]
        width = max(x_coords) - min(x_coords)
        height = max(y_coords) - min(y_coords)
        return width * height
    
    def calculate_distance(self, point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
        """Calculate Euclidean distance between two points"""
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def calculate_bbox_iou(self, bbox1: List[List[float]], bbox2: List[List[float]]) -> float:
        """Calculate IoU between two bboxes"""
        # Convert polygon bboxes to axis-aligned bounding boxes
        def poly_to_aabb(bbox):
            x_coords = [point[0] for point in bbox]
            y_coords = [point[1] for point in bbox]
            return [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]
        
        aabb1 = poly_to_aabb(bbox1)
        aabb2 = poly_to_aabb(bbox2)
        
        # Calculate intersection
        x1 = max(aabb1[0], aabb2[0])
        y1 = max(aabb1[1], aabb2[1])
        x2 = min(aabb1[2], aabb2[2])
        y2 = min(aabb1[3], aabb2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (aabb1[2] - aabb1[0]) * (aabb1[3] - aabb1[1])
        area2 = (aabb2[2] - aabb2[0]) * (aabb2[3] - aabb2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def calculate_cluster_center(self, cluster: List[Dict]) -> Tuple[float, float]:
        """Calculate center point of a cluster"""
        centers = [data['center'] for data in cluster]
        center_x = sum(c[0] for c in centers) / len(centers)
        center_y = sum(c[1] for c in centers) / len(centers)
        return (center_x, center_y)
    
    def calculate_cluster_bbox(self, cluster: List[Dict]) -> List[List[float]]:
        """Calculate bounding box that encompasses all bboxes in cluster"""
        all_points = []
        for data in cluster:
            all_points.extend(data['bbox'])
        
        x_coords = [point[0] for point in all_points]
        y_coords = [point[1] for point in all_points]
        
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)
        
        return [[min_x, min_y], [max_x, min_y], [max_x, max_y], [min_x, max_y]]
    
    def save_clusters(self, clusters: List[Dict], individual_results: List[Dict], csv_path: Path):
        """Save cluster and individual results to CSV file"""
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['id', 'type', 'bbox', 'text', 'score', 'bbox_count', 'texts'])
            writer.writeheader()
            
            # Write clusters
            for cluster in clusters:
                texts = [data['text'] for data in cluster['bboxes']]
                writer.writerow({
                    'id': f"cluster_{cluster['cluster_id']}",
                    'type': 'cluster',
                    'bbox': json.dumps(cluster['cluster_bbox']),
                    'text': '; '.join(texts),  # Combined text for clusters
                    'score': max(data['score'] for data in cluster['bboxes']),  # Highest score
                    'bbox_count': len(cluster['bboxes']),
                    'texts': json.dumps(texts)
                })
            
            # Write individual results
            for result in individual_results:
                writer.writerow({
                    'id': f"individual_{result['index']}",
                    'type': 'individual',
                    'bbox': json.dumps(result['bbox']),
                    'text': result['text'],
                    'score': result['score'],
                    'bbox_count': 1,
                    'texts': json.dumps([result['text']])
                })
    
    def create_visualization(self, image_path: Path, ocr_results: List[Dict], 
                           clusters: List[Dict], output_path: Path):
        """Create visualization of OCR results and clusters"""
        try:
            from PIL import ImageDraw, ImageFont
            
            # Load image
            img = Image.open(image_path)
            draw = ImageDraw.Draw(img)
            
            # Draw OCR bboxes in red
            for result in ocr_results:
                bbox = result['bbox']
                points = [(int(point[0]), int(point[1])) for point in bbox]
                draw.polygon(points, outline='red', width=2)
            
            # Draw cluster bboxes in blue
            for cluster in clusters:
                cluster_bbox = cluster['cluster_bbox']
                points = [(int(point[0]), int(point[1])) for point in cluster_bbox]
                draw.polygon(points, outline='blue', width=3)
            
            # Save visualization
            img.save(output_path)
            logger.info(f"Visualization saved to {output_path}")
            
        except Exception as e:
            logger.warning(f"Could not create visualization: {str(e)}")
    
    def create_final_csv(self, ocr_results: List[Dict], clusters: List[Dict], 
                        individual_results: List[Dict], post_ocr_results: Dict, output_path: Path):
        """Create final CSV with corrected results for both clusters and individual results"""
        try:
            final_results = []
            
            # Process clusters (with LLM corrections)
            if post_ocr_results and post_ocr_results.get('processed_clusters'):
                cluster_corrections = {}
                for cluster_result in post_ocr_results['processed_clusters']:
                    cluster_id = cluster_result['cluster_id']
                    cluster_type = cluster_result['cluster_type']
                    
                    if cluster_type == 'bbox' and cluster_result.get('merged_bbox'):
                        cluster_corrections[cluster_id] = {
                            'type': 'bbox',
                            'corrected_value': cluster_result['merged_bbox'],
                            'original_bboxes': cluster_result['original_bboxes']
                        }
                    elif cluster_type == 'label' and cluster_result.get('corrected_text'):
                        cluster_corrections[cluster_id] = {
                            'type': 'label',
                            'corrected_value': cluster_result['corrected_text'],
                            'original_texts': cluster_result['original_texts']
                        }
                
                # Add cluster results to final output
                for cluster in clusters:
                    cluster_id = cluster['cluster_id']
                    if cluster_id in cluster_corrections:
                        correction = cluster_corrections[cluster_id]
                        if correction['type'] == 'bbox':
                            # Bbox cluster - use merged bbox
                            final_results.append({
                                'id': f"cluster_{cluster_id}",
                                'type': 'cluster',
                                'original_bbox': json.dumps(cluster['cluster_bbox']),
                                'corrected_bbox': correction['corrected_value'],
                                'text': '; '.join([data['text'] for data in cluster['bboxes']]),
                                'corrected_text': '; '.join([data['text'] for data in cluster['bboxes']]),
                                'score': max(data['score'] for data in cluster['bboxes']),
                                'correction_type': 'bbox_merge',
                                'bbox_count': len(cluster['bboxes'])
                            })
                        elif correction['type'] == 'label':
                            # Label cluster - use corrected text
                            final_results.append({
                                'id': f"cluster_{cluster_id}",
                                'type': 'cluster',
                                'original_bbox': json.dumps(cluster['cluster_bbox']),
                                'corrected_bbox': json.dumps(cluster['cluster_bbox']),
                                'text': '; '.join([data['text'] for data in cluster['bboxes']]),
                                'corrected_text': correction['corrected_value'],
                                'score': max(data['score'] for data in cluster['bboxes']),
                                'correction_type': 'text_correction',
                                'bbox_count': len(cluster['bboxes'])
                            })
            
            # Process individual results (no LLM processing needed)
            for result in individual_results:
                final_results.append({
                    'id': f"individual_{result['index']}",
                    'type': 'individual',
                    'original_bbox': json.dumps(result['bbox']),
                    'corrected_bbox': json.dumps(result['bbox']),  # No correction
                    'text': result['text'],
                    'corrected_text': result['text'],  # No correction
                    'score': result['score'],
                    'correction_type': 'none',
                    'bbox_count': 1
                })
            
            # Save final CSV
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                if final_results:
                    writer = csv.DictWriter(f, fieldnames=[
                        'id', 'type', 'original_bbox', 'corrected_bbox', 'text', 'corrected_text', 
                        'score', 'correction_type', 'bbox_count'
                    ])
                    writer.writeheader()
                    for result in final_results:
                        writer.writerow(result)
            
            logger.info(f"Final results saved to {output_path}")
            logger.info(f"Total results: {len(final_results)} (clusters: {len(clusters)}, individual: {len(individual_results)})")
            
        except Exception as e:
            logger.error(f"Error creating final CSV: {str(e)}")
    
    def run_pipeline(self):
        """Run the complete pipeline on all images"""
        logger.info("Starting OCR and Post-OCR Pipeline")
        logger.info(f"Input directory: {self.input_dir}")
        logger.info(f"Output directory: {self.output_dir}")
        
        # Get all image files
        image_files = self.get_image_files()
        
        if not image_files:
            logger.warning("No image files found in input directory")
            return
        
        logger.info(f"Found {len(image_files)} images to process")
        
        # Process each image
        results = []
        for image_path in image_files:
            result = self.process_single_image(image_path)
            results.append(result)
        
        # Save summary
        summary_path = self.output_dir / 'processing_summary.json'
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # Print summary
        successful = sum(1 for r in results if r['status'] == 'success')
        failed = len(results) - successful
        
        logger.info(f"Pipeline completed!")
        logger.info(f"Successfully processed: {successful} images")
        logger.info(f"Failed: {failed} images")
        logger.info(f"Summary saved to: {summary_path}")


def main():
    """Main function to run the pipeline"""
    parser = argparse.ArgumentParser(description='OCR and Post-OCR Pipeline')
    parser.add_argument('--input', '-i', default='input', 
                       help='Input directory containing images (default: input)')
    parser.add_argument('--output', '-o', default='output', 
                       help='Output directory for results (default: output)')
    
    args = parser.parse_args()
    
    # Create and run pipeline
    pipeline = OCRPostOCRPipeline(args.input, args.output)
    pipeline.run_pipeline()


if __name__ == "__main__":
    main()
