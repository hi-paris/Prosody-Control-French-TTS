import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from tqdm import tqdm
import re
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_model_and_tokenizer(model_name="jonahdvt/qwen-z2y-lora"):
    """Loads the fine-tuned LoRA model and its tokenizer from Hugging Face Hub."""
    logger.info(f"Loading base model and tokenizer for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B")
    base_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-7B",
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    logger.info(f"Loading LoRA adapter from {model_name}...")
    model = PeftModel.from_pretrained(base_model, model_name)
    model.eval() # Set to evaluation mode
    return model, tokenizer

def load_test_data(file_path):
    """Loads test data from a JSON file."""
    logger.info(f"Loading test data from {file_path}...")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    logger.info(f"Loaded {len(data)} test examples.")
    return data

def compute_calibration_stats(test_data):
    """
    Compute global mean and std for prosody parameters from all test data
    to enable z-score normalization like in BiLSTM script.
    """
    all_prosody = []
    
    for item in test_data:
        ground_truth_y = item['y']
        ref_params = extract_ssml_parameters(ground_truth_y)
        
        # Get all prosody values (pitch, volume, rate) - excluding break_time
        for pitch in ref_params['pitch']:
            all_prosody.append([pitch, 0.0, 0.0])  # placeholder for volume, rate
        for volume in ref_params['volume']:
            if all_prosody:
                all_prosody[-1][1] = volume
        for rate in ref_params['rate']:
            if all_prosody:
                all_prosody[-1][2] = rate
                
        # If we have individual prosody elements, add them
        prosody_blocks = re.findall(r'<prosody([^>]*?)>(.*?)</prosody>', ground_truth_y, re.DOTALL)
        for block_attrs, _ in prosody_blocks:
            pitch_val, volume_val, rate_val = 0.0, 0.0, 0.0
            
            pitch_match = re.search(r'pitch="([+-]?\d+\.?\d*)%?"', block_attrs)
            if pitch_match:
                pitch_val = float(pitch_match.group(1))
                
            volume_match = re.search(r'volume="([+-]?\d+\.?\d*)%?"', block_attrs)
            if volume_match:
                volume_val = float(volume_match.group(1))
                
            rate_match = re.search(r'rate="([+-]?\d+\.?\d*)%?"', block_attrs)
            if rate_match:
                rate_val = float(rate_match.group(1))
                
            all_prosody.append([pitch_val, volume_val, rate_val])
    
    if not all_prosody:
        # Default values if no prosody found
        return np.array([0.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0])
    
    arr = np.array(all_prosody, dtype=np.float32)
    mean = arr.mean(axis=0)
    std = arr.std(axis=0)
    std[std == 0] = 1.0  # Avoid division by zero
    
    logger.info(f"Calibration stats - Mean: {mean}, Std: {std}")
    return mean, std

def extract_ssml_parameters(ssml_text):
    """
    Extracts break time, pitch, volume, and rate values from SSML text.
    Returns lists of numerical values for each parameter.
    Handles various formats (e.g., "500ms", "-1.60%", "-10.00").
    """
    params = {
        'break_time': [],
        'pitch': [],
        'volume': [],
        'rate': []
    }

    # Regex for break time (e.g., <break time="500ms"/>)
    # The '?' makes 'ms' optional, in case it's just a number
    break_matches = re.findall(r'<break time="(\d+)ms?"', ssml_text)
    params['break_time'].extend([float(m) for m in break_matches])

    # Regex for prosody attributes (pitch, rate, volume)
    # It captures the entire prosody block and then extracts attributes from it
    prosody_blocks = re.findall(r'<prosody([^>]*)>', ssml_text)
    for block in prosody_blocks:
        # Extract pitch
        pitch_match = re.search(r'pitch="([+-]?\d+\.?\d*)%?"', block)
        if pitch_match:
            params['pitch'].append(float(pitch_match.group(1)))
        
        # Extract rate
        rate_match = re.search(r'rate="([+-]?\d+\.?\d*)%?"', block)
        if rate_match:
            params['rate'].append(float(rate_match.group(1)))
            
        # Extract volume
        volume_match = re.search(r'volume="([+-]?\d+\.?\d*)%?"', block)
        if volume_match:
            params['volume'].append(float(volume_match.group(1)))
            
    return params

def normalize_prosody_params(params, mean, std):
    """
    Apply z-score normalization to prosody parameters (pitch, volume, rate)
    using the global statistics, similar to BiLSTM approach.
    """
    normalized_params = {
        'break_time': params['break_time'].copy(),  # Don't normalize break_time
        'pitch': [],
        'volume': [],
        'rate': []
    }
    
    # Normalize pitch, volume, rate using z-score
    for i, pitch in enumerate(params['pitch']):
        normalized_params['pitch'].append((pitch - mean[0]) / std[0])
    
    for i, volume in enumerate(params['volume']):
        normalized_params['volume'].append((volume - mean[1]) / std[1])
        
    for i, rate in enumerate(params['rate']):
        normalized_params['rate'].append((rate - mean[2]) / std[2])
    
    return normalized_params

def calculate_regression_metrics(predictions_params, references_params, mean, std):
    """
    Calculates MSE, MAE, RMSE, and R^2 for each parameter.
    Uses both raw and normalized metrics for comparison with BiLSTM.
    """
    metrics = {}
    
    # Calculate metrics for break_time (not normalized)
    param_name = 'break_time'
    pred_values = np.array(predictions_params[param_name])
    ref_values = np.array(references_params[param_name])
    
    if len(pred_values) > 0 and len(ref_values) > 0:
        min_len = min(len(pred_values), len(ref_values))
        pred_values = pred_values[:min_len]
        ref_values = ref_values[:min_len]
        
        mse = mean_squared_error(ref_values, pred_values)
        mae = mean_absolute_error(ref_values, pred_values)
        rmse = np.sqrt(mse)
        r2 = r2_score(ref_values, pred_values) if len(pred_values) > 1 else float('nan')
        
        metrics[param_name] = {
            "mse": mse, "mae": mae, "rmse": rmse, "r2": r2,
            "count_pred": len(pred_values), "count_ref": len(ref_values)
        }
    else:
        metrics[param_name] = {
            "mse": float('nan'), "mae": float('nan'), "rmse": float('nan'), "r2": float('nan'),
            "count_pred": len(pred_values), "count_ref": len(ref_values)
        }
    
    # Calculate metrics for prosody parameters (pitch, volume, rate) - both raw and normalized
    prosody_params = ['pitch', 'volume', 'rate']
    for i, param_name in enumerate(prosody_params):
        pred_values = np.array(predictions_params[param_name])
        ref_values = np.array(references_params[param_name])
        
        if len(pred_values) > 0 and len(ref_values) > 0:
            min_len = min(len(pred_values), len(ref_values))
            pred_values = pred_values[:min_len]
            ref_values = ref_values[:min_len]
            
            # Raw metrics
            mse_raw = mean_squared_error(ref_values, pred_values)
            mae_raw = mean_absolute_error(ref_values, pred_values)
            rmse_raw = np.sqrt(mse_raw)
            r2_raw = r2_score(ref_values, pred_values) if len(pred_values) > 1 else float('nan')
            
            # Normalized metrics (z-score)
            pred_norm = (pred_values - mean[i]) / std[i]
            ref_norm = (ref_values - mean[i]) / std[i]
            
            mse_norm = mean_squared_error(ref_norm, pred_norm)
            mae_norm = mean_absolute_error(ref_norm, pred_norm)
            rmse_norm = np.sqrt(mse_norm)
            r2_norm = r2_score(ref_norm, pred_norm) if len(pred_norm) > 1 else float('nan')
            
            metrics[param_name] = {
                "mse_raw": mse_raw, "mae_raw": mae_raw, "rmse_raw": rmse_raw, "r2_raw": r2_raw,
                "mse_norm": mse_norm, "mae_norm": mae_norm, "rmse_norm": rmse_norm, "r2_norm": r2_norm,
                "count_pred": len(pred_values), "count_ref": len(ref_values)
            }
        else:
            metrics[param_name] = {
                "mse_raw": float('nan'), "mae_raw": mae_raw, "rmse_raw": float('nan'), "r2_raw": float('nan'),
                "mse_norm": float('nan'), "mae_norm": float('nan'), "rmse_norm": float('nan'), "r2_norm": float('nan'),
                "count_pred": len(pred_values), "count_ref": len(ref_values)
            }
    
    return metrics

def main():
    """Main function to load model, test, and calculate metrics."""
    model_name = "jonahdvt/qwen-z2y-lora"
    test_data_path = "/home/mila/d/dauvetj/mon_projet_TTS/Code/ssml_models/jonah/full_data_xyz/test.json"

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_name)

    # Load test data
    test_data = load_test_data(test_data_path)
    
    # Compute calibration statistics for z-score normalization
    mean, std = compute_calibration_stats(test_data)

    all_predictions_params = { 'break_time': [], 'pitch': [], 'volume': [], 'rate': [] }
    all_references_params = { 'break_time': [], 'pitch': [], 'volume': [], 'rate': [] }

    # Instruction for the z to y task
    instruction = "Fill in the SSML parameters (time, pitch, volume, rate):"

    logger.info("Generating predictions...")
    for i, item in enumerate(tqdm(test_data, desc="Generating SSML parameters")):
        input_z = item['z'] # Input is 'z'
        ground_truth_y = item['y'] # Target is 'y'

        # Format input for the model
        formatted_input = f"### Task:\n{instruction}\n\n### Z_SSML:\n{input_z}\n\n### Y_SSML:\n"

        inputs = tokenizer(formatted_input, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512, # Increased max_new_tokens for potentially longer SSML
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=False
            )
        
        generated_token_ids = outputs.sequences[0]
        generated_ssml_raw = tokenizer.decode(generated_token_ids, skip_special_tokens=True)
        
        # Extract the relevant SSML part from the generated text
        ssml_output_start_idx = generated_ssml_raw.find("### Y_SSML:\n")
        if ssml_output_start_idx != -1:
            extracted_ssml = generated_ssml_raw[ssml_output_start_idx + len("### Y_SSML:\n"):].strip()
        else:
            extracted_ssml = generated_ssml_raw.strip() # Fallback if format is not perfect

        # --- Print outputs as they come ---
        print(f"\n--- Example {i+1} ---")
        print(f"Input Z: \n{input_z}")
        print(f"Ground Truth Y: \n{ground_truth_y}")
        print(f"Generated Y: \n{extracted_ssml}")
        print("---------------------\n")
        # --- End print outputs ---

        # Extract parameters for this example
        pred_params = extract_ssml_parameters(extracted_ssml)
        ref_params = extract_ssml_parameters(ground_truth_y)

        # Aggregate parameters for overall metric calculation
        for param_name in all_predictions_params.keys():
            all_predictions_params[param_name].extend(pred_params[param_name])
            all_references_params[param_name].extend(ref_params[param_name])
            
    logger.info("Calculating regression metrics...")
    metrics = calculate_regression_metrics(all_predictions_params, all_references_params, mean, std)

    print("\n--- Evaluation Results ---")
    print("(Raw metrics are on original scale, Normalized metrics are z-score normalized like BiLSTM)")
    print()
    
    # Print break_time metrics (only raw, not normalized)
    print("--- Break Time Metrics ---")
    param_metrics = metrics['break_time']
    for metric_name, value in param_metrics.items():
        if 'count' in metric_name:
            print(f"  {metric_name}: {value}")
        else:
            print(f"  {metric_name.upper()}: {value:.4f}")
    print("-" * 30)
    
    # Print prosody metrics (both raw and normalized)
    prosody_params = ['pitch', 'volume', 'rate']
    for param_name in prosody_params:
        print(f"--- {param_name.replace('_', ' ').title()} Metrics ---")
        param_metrics = metrics[param_name]
        
        print("  RAW METRICS:")
        for metric_name, value in param_metrics.items():
            if 'raw' in metric_name:
                base_name = metric_name.replace('_raw', '').upper()
                if not np.isnan(value):
                    print(f"    {base_name}: {value:.4f}")
                else:
                    print(f"    {base_name}: NaN")
        
        print("  NORMALIZED METRICS (comparable to BiLSTM):")
        for metric_name, value in param_metrics.items():
            if 'norm' in metric_name:
                base_name = metric_name.replace('_norm', '').upper()
                if not np.isnan(value):
                    print(f"    {base_name}: {value:.4f}")
                else:
                    print(f"    {base_name}: NaN")
        
        print(f"  COUNT_PRED: {param_metrics['count_pred']}")
        print(f"  COUNT_REF: {param_metrics['count_ref']}")
        print("-" * 30)
    
    # Summary comparable to BiLSTM
    print("\n--- SUMMARY (Normalized MSE - Comparable to BiLSTM) ---")
    norm_mses = []
    for param_name in prosody_params:
        mse_norm = metrics[param_name]['mse_norm']
        if not np.isnan(mse_norm):
            print(f"  {param_name.title()}: {mse_norm:.4f}")
            norm_mses.append(mse_norm)
        else:
            print(f"  {param_name.title()}: NaN")
    
    if norm_mses:
        avg_mse = np.mean(norm_mses)
        print(f"  Average: {avg_mse:.4f}")
    else:
        print("  Average: NaN")
    print("-" * 50)

if __name__ == "__main__":
    main()