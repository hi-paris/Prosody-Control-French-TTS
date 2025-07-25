import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from tqdm import tqdm
import re
import numpy as np
import logging
import string

# Configuration
NORMALIZED = True  # Set to True to strip all punctuation and capitalization from input text

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def normalize_text(text):
    """Strips all punctuation and capitalization from text if NORMALIZED is True."""
    if not NORMALIZED:
        return text
    
    # Remove all punctuation and convert to lowercase
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.lower()
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text

def load_model_and_tokenizer(model_name="jonahdvt/qwen-ssml-lora"):
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

def preprocess_ssml(ssml_text):
    """Normalizes SSML for consistent comparison."""
    # Remove extra spaces around tags
    ssml_text = re.sub(r'\s*<break\s*/>\s*', '<break/>', ssml_text)
    # Remove leading/trailing whitespace
    return ssml_text.strip()

def calculate_metrics(predictions, references):
    """Calculates Exact Match, SSML Tag Accuracy, and Perplexity."""
    exact_matches = 0
    
    # For perplexity
    losses = []
    
    loss_fct = torch.nn.CrossEntropyLoss(reduction='mean')

    f1_scores = [] # Store F1 scores for each example's breaks

    for i, (pred_ssml, ref_ssml, pred_tensor, ref_tensor) in enumerate(zip(
        predictions['generated_ssml'], 
        references['original_ssml'], 
        predictions['raw_output_tensors'], 
        references['raw_label_tensors']
    )):
        # Exact Match
        if preprocess_ssml(pred_ssml) == preprocess_ssml(ref_ssml):
            exact_matches += 1

        # SSML Tag Accuracy (F1 on break tags)
        # Extract break positions for a more robust F1
        # This is a simplified approach. For true F1, you'd need to tokenize text and breaks.
        # Here, we'll just count them as set of positions relative to the start of the string, or simply count them.
        
        # Let's count the breaks for a simple F1 based on break presence/absence per example
        pred_breaks_count = pred_ssml.count('<break/>')
        ref_breaks_count = ref_ssml.count('<break/>')

        # This F1 is for whether an example "has breaks" correctly, not about exact positions.
        # A more advanced F1 would compare sets of break *positions* or use sequence matching.
        
        tp = 0
        fp = 0
        fn = 0
        
        # If both have breaks and the count is close (within a small threshold) it could be considered a "hit"
        # For simplicity, let's consider a TP if both have breaks AND the counts are reasonably close.
        # Or, just if both have breaks.
        
        # For a more robust F1, we need to consider if a break was *correctly placed*.
        # This means comparing the original text between breaks.
        # Example: "Hello.<break/>How are you?" vs "Hello.<break/>How<break/>are you?"
        # The current simple F1 for tag presence is: did the model decide to add breaks vs not add them.

        # Let's adjust F1 to be based on the *presence* of breaks in the generated vs reference.
        # True Positive: model predicted breaks AND reference has breaks.
        # False Positive: model predicted breaks BUT reference has no breaks.
        # False Negative: model predicted no breaks BUT reference has breaks.
        
        if pred_breaks_count > 0 and ref_breaks_count > 0:
            tp = 1
        elif pred_breaks_count > 0 and ref_breaks_count == 0:
            fp = 1
        elif pred_breaks_count == 0 and ref_breaks_count > 0:
            fn = 1
        
        if (tp + fp) == 0:
            precision = 0
        else:
            precision = tp / (tp + fp)
            
        if (tp + fn) == 0:
            recall = 0
        else:
            recall = tp / (tp + fn)
            
        if (precision + recall) == 0:
            f1 = 0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)
        f1_scores.append(f1)

        # Perplexity Calculation
        # Only consider positions where labels are not -100
        # The pred_tensor here contains logits for the *reference* sequence (from the model pass on ground truth)
        # ref_tensor contains the labels (ground truth token IDs with instruction masked)
        
        shift_logits = pred_tensor[..., :-1, :].contiguous()
        shift_labels = ref_tensor[..., 1:].contiguous()
        
        # Flatten the tokens
        loss_labels = shift_labels.view(-1)
        loss_logits = shift_logits.view(-1, shift_logits.size(-1))

        # Only compute loss over non-masked tokens
        active_loss = loss_labels != -100
        if active_loss.sum() > 0:
            current_loss = loss_fct(loss_logits[active_loss], loss_labels[active_loss])
            losses.append(current_loss.item())

    total_examples = len(references['original_ssml'])
    exact_match_score = exact_matches / total_examples if total_examples > 0 else 0

    ssml_tag_accuracy_f1 = np.mean(f1_scores) if f1_scores else 0 # Average F1 across examples

    perplexity = np.exp(np.mean(losses)) if losses else float('inf')

    return {
        "exact_match_score": exact_match_score,
        "ssml_tag_accuracy_f1": ssml_tag_accuracy_f1, # Simplified F1 for break presence
        "perplexity": perplexity
    }

def main():
    """Main function to load model, test, and calculate metrics."""
    model_name = "jonahdvt/qwen-ssml-lora"
    test_data_path = "/home/mila/d/dauvetj/mon_projet_TTS/Code/ssml_models/jonah/full_data/test.json"

    logger.info(f"Text normalization mode: {'ENABLED' if NORMALIZED else 'DISABLED'}")
    if NORMALIZED:
        logger.info("Input text will be stripped of all punctuation and capitalization")

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_name)

    # Load test data
    test_data = load_test_data(test_data_path)

    predictions = {
        'generated_ssml': [],
        'raw_output_tensors': [] # Stores logits from model pass on reference
    }
    references = {
        'original_ssml': [],
        'raw_label_tensors': [] # Stores masked labels for reference
    }

    instruction = "Convert text to SSML with pauses:"

    logger.info("Generating predictions...")
    for i, item in enumerate(tqdm(test_data, desc="Generating SSML")):
        input_text = item['x']
        ground_truth_ssml = item['y']

        # Apply normalization if enabled
        processed_input_text = normalize_text(input_text)

        formatted_input = f"### Task:\n{instruction}\n\n### Text:\n{processed_input_text}\n\n### SSML:\n"

        inputs = tokenizer(formatted_input, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256, # Adjust as needed
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=False # Not directly needed for perplexity if we re-pass ref
            )
        
        generated_token_ids = outputs.sequences[0]
        generated_ssml_raw = tokenizer.decode(generated_token_ids, skip_special_tokens=True)
        
        # Extract the SSML part from the generated text
        ssml_output_start_idx = generated_ssml_raw.find("### SSML:\n")
        if ssml_output_start_idx != -1:
            extracted_ssml = generated_ssml_raw[ssml_output_start_idx + len("### SSML:\n"):].strip()
        else:
            extracted_ssml = generated_ssml_raw.strip() # Fallback if format is not perfect

        predictions['generated_ssml'].append(extracted_ssml)
        
        # --- Print outputs as they come ---
        print(f"\n--- Example {i+1} ---")
        print(f"Original Input Text: {input_text}")
        if NORMALIZED:
            print(f"Normalized Input Text: {processed_input_text}")
        print(f"Ground Truth SSML: {ground_truth_ssml}")
        print(f"Generated SSML: {extracted_ssml}")
        print("---------------------\n")
        # --- End print outputs ---

        # For perplexity calculation, we need to get the model's logits for the *ground truth* sequence.
        # This is because perplexity is a measure of how well the model predicts the *actual* sequence.
        # Note: We use the processed input text for consistency with what was used for generation
        full_input_and_target = formatted_input + ground_truth_ssml
        tokenized_ref_for_loss = tokenizer(
            full_input_and_target, 
            return_tensors="pt", 
            truncation=True, 
            max_length=tokenizer.model_max_length # Use model_max_length for consistency
        )
        
        input_ids_ref_for_loss = tokenized_ref_for_loss['input_ids'].to(model.device)
        attention_mask_ref_for_loss = tokenized_ref_for_loss['attention_mask'].to(model.device)
        
        with torch.no_grad():
            # Pass the ground truth sequence through the model to get logits
            ref_outputs_for_loss = model(
                input_ids=input_ids_ref_for_loss, 
                attention_mask=attention_mask_ref_for_loss, 
                labels=input_ids_ref_for_loss # Labels are needed for loss calculation
            )
            
        references['original_ssml'].append(ground_truth_ssml)
        
        predictions['raw_output_tensors'].append(ref_outputs_for_loss.logits)
        
        # Mask labels correctly for the reference pass (for loss calculation)
        ref_labels_masked = input_ids_ref_for_loss.clone()
        instruction_length_ref = len(tokenizer(formatted_input, add_special_tokens=False)['input_ids'])
        # Ensure instruction length does not exceed sequence length
        if instruction_length_ref < ref_labels_masked.shape[1]:
            ref_labels_masked[0, :instruction_length_ref] = -100
        
        references['raw_label_tensors'].append(ref_labels_masked)


    logger.info("Calculating metrics...")
    metrics = calculate_metrics(predictions, references)

    print("\n--- Evaluation Results ---")
    print(f"Text normalization: {'ENABLED' if NORMALIZED else 'DISABLED'}")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    print("--------------------------")

if __name__ == "__main__":
    main()