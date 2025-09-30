from unsloth import FastLanguageModel, is_bfloat16_supported
import torch
import json
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

class PostOCRProcessor:
    """Post-OCR processor using LLM with LoRA models for bbox and label processing"""
    
    def __init__(self, model_name: str = "unsloth/Qwen2.5-3B-Instruct", 
                 max_seq_length: int = 4096, lora_rank: int = 64):
        self.model_name = model_name
        self.max_seq_length = max_seq_length
        self.lora_rank = lora_rank
        
        # Initialize model and tokenizer
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            load_in_4bit=True,  # False for LoRA 16bit
            fast_inference=True,  # Enable vLLM fast inference
            max_lora_rank=lora_rank,
            gpu_memory_utilization=0.8,  # Reduce if out of memory
        )

        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=lora_rank,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],  # Remove QKVO if out of memory
            lora_alpha=lora_rank,
            use_gradient_checkpointing="unsloth",  # Enable long context finetuning
            random_state=3407,
        )
        
        # Load LoRA models
        self.bboxes_lora_path = Path(__file__).parent / "grpo_model" / "bboxes"
        self.labels_lora_path = Path(__file__).parent / "grpo_model" / "labels"

    def get_system_prompt(self, task_type: str = "bbox") -> str:
        """Get system prompt based on task type"""
        if task_type == "bbox":
            return """
            You are an expert at analyzing and merging bounding boxes from OCR results.
            Your task is to merge clusters of overlapping or nearby bounding boxes into single, optimal bounding boxes.
            Consider the spatial relationships and text content when making decisions.
            
            Respond in the following format:
            <reasoning>
            Analyze the spatial relationships and text content of the bboxes...
            </reasoning>
            <answer>
            [merged_bbox_coordinates]
            </answer>
            """
        else:  # labels
            return """
            You are an expert at analyzing and correcting text from OCR results.
            Your task is to correct, merge, and improve the text content from OCR detections.
            Consider the context and spatial relationships when making corrections.
            
            Respond in the following format:
            <reasoning>
            Analyze the text content and context...
            </reasoning>
            <answer>
            [corrected_text]
            </answer>
            """
    
    def process_clusters(self, clusters: List[Dict], ocr_results: List[Dict], 
                        output_path: str) -> Dict[str, Any]:
        """Process clusters using appropriate LoRA model"""
        results = {
            'processed_clusters': [],
            'total_clusters': len(clusters),
            'bbox_clusters': 0,
            'label_clusters': 0
        }
        
        for cluster in clusters:
            cluster_id = cluster['cluster_id']
            bboxes = cluster['bboxes']
            
            # Determine if this is a bbox cluster or label cluster
            # Simple heuristic: if cluster has multiple bboxes, it's a bbox cluster
            is_bbox_cluster = len(bboxes) > 1
            
            if is_bbox_cluster:
                result = self.process_bbox_cluster(cluster)
                results['bbox_clusters'] += 1
            else:
                result = self.process_label_cluster(cluster)
                results['label_clusters'] += 1
            
            result['cluster_id'] = cluster_id
            result['cluster_type'] = 'bbox' if is_bbox_cluster else 'label'
            results['processed_clusters'].append(result)
        
        # Save results
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Post-OCR processing completed: {results['bbox_clusters']} bbox clusters, {results['label_clusters']} label clusters")
        return results
    
    def process_bbox_cluster(self, cluster: Dict) -> Dict[str, Any]:
        """Process a bbox cluster using bboxes LoRA model"""
        bboxes = cluster['bboxes']
        bbox_coords = [data['bbox'] for data in bboxes]
        
        # Create prompt for bbox merging
        prompt = f"Merging cluster of bboxes: {bbox_coords} to one bbox"
        
        # Use bboxes LoRA model
        lora_path = str(self.bboxes_lora_path)
        
        try:
            raw_result = self.generate_with_lora(prompt, lora_path, task_type="bbox")
            parsed_result = self.parse_llm_response(raw_result, task_type="bbox")
            return {
                'original_bboxes': bbox_coords,
                'merged_bbox': parsed_result,
                'raw_llm_response': raw_result,
                'texts': [data['text'] for data in bboxes]
            }
        except Exception as e:
            logger.error(f"Error processing bbox cluster: {str(e)}")
            return {
                'original_bboxes': bbox_coords,
                'merged_bbox': None,
                'error': str(e),
                'texts': [data['text'] for data in bboxes]
            }
    
    def process_label_cluster(self, cluster: Dict) -> Dict[str, Any]:
        """Process a label cluster using labels LoRA model"""
        bboxes = cluster['bboxes']
        texts = [data['text'] for data in bboxes]
        
        # Create prompt for text correction
        prompt = f"Correct and merge the following OCR texts: {texts}"
        
        # Use labels LoRA model
        lora_path = str(self.labels_lora_path)
        
        try:
            raw_result = self.generate_with_lora(prompt, lora_path, task_type="label")
            parsed_result = self.parse_llm_response(raw_result, task_type="label")
            return {
                'original_texts': texts,
                'corrected_text': parsed_result,
                'raw_llm_response': raw_result,
                'bboxes': [data['bbox'] for data in bboxes]
            }
        except Exception as e:
            logger.error(f"Error processing label cluster: {str(e)}")
            return {
                'original_texts': texts,
                'corrected_text': None,
                'error': str(e),
                'bboxes': [data['bbox'] for data in bboxes]
            }
    
    def generate_with_lora(self, prompt: str, lora_path: str, task_type: str = "bbox") -> str:
        """Generate response using LoRA model"""
        from vllm import SamplingParams
        
        system_prompt = self.get_system_prompt(task_type)
        
        text = self.tokenizer.apply_chat_template([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ], tokenize=False, add_generation_prompt=True)
        
        sampling_params = SamplingParams(
            temperature=0.8,
            top_p=0.95,
            max_tokens=1024,
        )
        
        output = self.model.fast_generate(
            text,
            sampling_params=sampling_params,
            lora_request=self.model.load_lora(lora_path),
        )[0].outputs[0].text
        
        return output
    
    def parse_llm_response(self, response: str, task_type: str = "bbox") -> str:
        """Parse LLM response to extract the actual result"""
        try:
            if "<answer>" in response and "</answer>" in response:
                # Extract content between <answer> tags
                start = response.find("<answer>") + len("<answer>")
                end = response.find("</answer>")
                answer = response[start:end].strip()
                
                if task_type == "bbox":
                    # Parse bbox coordinates
                    return self.parse_bbox_coordinates(answer)
                else:
                    # Return text as is for labels
                    return answer
            else:
                # If no structured response, return the whole response
                logger.warning(f"No structured answer found in response: {response[:100]}...")
                return response
        except Exception as e:
            logger.error(f"Error parsing LLM response: {str(e)}")
            return response
    
    def parse_bbox_coordinates(self, bbox_str: str) -> str:
        """Parse bbox coordinates from LLM response"""
        try:
            # Remove extra quotes and clean up
            bbox_str = bbox_str.strip().strip("'\"")
            
            # Try to extract coordinates from various formats
            import re
            
            # Look for coordinate patterns like (x1, y1, x2, y2) or [x1, y1, x2, y2]
            coord_pattern = r'[\[\(](\d+),\s*(\d+),\s*(\d+),\s*(\d+)[\]\)]'
            match = re.search(coord_pattern, bbox_str)
            
            if match:
                x1, y1, x2, y2 = map(int, match.groups())
                return f"({x1}, {y1}, {x2}, {y2})"
            else:
                # If no standard format found, return as is
                return bbox_str
                
        except Exception as e:
            logger.error(f"Error parsing bbox coordinates: {str(e)}")
            return bbox_str
