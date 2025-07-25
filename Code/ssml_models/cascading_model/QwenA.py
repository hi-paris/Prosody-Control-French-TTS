
#!/usr/bin/env python3
"""
Optimized Qwen 7B Training Script for Text-to-SSML Pause Prediction
Memory-optimized version with DeepSpeed, LoRA, and other optimizations
"""

import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from datasets import Dataset as HFDataset
import logging
from datetime import datetime
import numpy as np
from tqdm import tqdm
import gc
import torch.nn as nn

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SSMLDataset(Dataset):
    """Memory-optimized dataset class for text-to-SSML conversion"""
    
    def __init__(self, data_path, tokenizer, max_length=1024):  # Reduced max length
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = self.load_data(data_path)
        
    def load_data(self, data_path):
        """Load and preprocess the training data"""
        logger.info(f"Loading data from {data_path}")
        
        all_data = []
        
        # Load all JSON files in the data directory
        if os.path.isdir(data_path):
            for filename in os.listdir(data_path):
                if filename.endswith('.json'):
                    file_path = os.path.join(data_path, filename)
                    with open(file_path, 'r', encoding='utf-8') as f:
                        try:
                            data = json.load(f)
                            if isinstance(data, list):
                                all_data.extend(data)
                            else:
                                all_data.append(data)
                        except json.JSONDecodeError as e:
                            logger.warning(f"Error loading {filename}: {e}")
        else:
            # Single file
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    all_data = data
                else:
                    all_data = [data]
        
        # Filter out very long examples to save memory
        filtered_data = []
        for item in all_data:
            input_text = item['x']
            target_text = item['y']
            
            # Rough estimate of token count
            estimated_tokens = len(input_text.split()) + len(target_text.split())
            if estimated_tokens * 1.3 < self.max_length * 0.8:  # Conservative estimate
                filtered_data.append(item)
        
        logger.info(f"Loaded {len(filtered_data)} training examples (filtered from {len(all_data)})")
        return filtered_data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Format the input and output
        input_text = item['x']
        target_text = item['y']
        
        # Create instruction format (shorter prompt to save tokens)
        instruction = "Convert text to SSML with pauses:"
        formatted_input = f"### Task:\n{instruction}\n\n### Text:\n{input_text}\n\n### SSML:\n"
        full_text = formatted_input + target_text
        
        # Tokenize with stricter length control
        tokenized = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors="pt"
        )
        
        # Create labels (same as input_ids for causal LM)
        labels = tokenized['input_ids'].clone()
        
        # Mask the instruction part so we only compute loss on the response
        instruction_length = len(self.tokenizer(formatted_input, add_special_tokens=False)['input_ids'])
        labels[0, :instruction_length] = -100
        
        # Return tensors directly (not squeezed) to avoid conversion issues
        return {
            'input_ids': tokenized['input_ids'].squeeze().long(),
            'attention_mask': tokenized['attention_mask'].squeeze().long(),
            'labels': labels.squeeze().long()
        }

def setup_cache_directories():
    """Setup cache directories based on available environment variables"""
    cache_dir = None
    
    # Priority: SLURM_TMPDIR > SCRATCH > HOME
    if os.environ.get('SLURM_TMPDIR'):
        cache_dir = os.path.join(os.environ['SLURM_TMPDIR'], 'hf_cache')
        logger.info(f"Using SLURM_TMPDIR for cache: {cache_dir}")
    elif os.environ.get('SCRATCH'):
        cache_dir = os.path.join(os.environ['SCRATCH'], 'hf_cache')
        logger.info(f"Using SCRATCH for cache: {cache_dir}")
    else:
        cache_dir = os.path.expanduser('~/.cache/huggingface')
        logger.info(f"Using HOME cache: {cache_dir}")
    
    os.makedirs(cache_dir, exist_ok=True)
    
    # Set environment variables
    os.environ['HF_HOME'] = cache_dir
    os.environ['TRANSFORMERS_CACHE'] = cache_dir
    os.environ['HF_DATASETS_CACHE'] = cache_dir
    
    return cache_dir

def load_model_and_tokenizer(model_name="Qwen/Qwen2.5-7B", cache_dir=None, use_lora=True):
    """Load the Qwen model and tokenizer with memory optimizations"""
    logger.info(f"Loading model: {model_name}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        trust_remote_code=True
    )
    
    # Add special tokens if needed
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with optimizations
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        # Memory optimizations
        low_cpu_mem_usage=True,
        max_memory={0: "32GB"},  # More conservative memory limit
    )
    
    # Prepare model for LoRA training
    if use_lora:
        # Enable gradient checkpointing first
        model.gradient_checkpointing_enable()
        
        # Prepare model for k-bit training (handles gradient setup properly)
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        
        # LoRA configuration - targeting the right modules for Qwen2.5
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=8,  # Reasonable rank for good performance
            lora_alpha=16,
            lora_dropout=0.1,
            target_modules=[
                "q_proj",
                "k_proj", 
                "v_proj",
                "o_proj",
                "gate_proj",  # Important for Qwen architecture
                "up_proj",    # Important for Qwen architecture
                "down_proj"   # Important for Qwen architecture
            ],
            bias="none",
            modules_to_save=None,  # Don't save additional modules
        )
        
        model = get_peft_model(model, lora_config)
        
        # Ensure model is in training mode
        model.train()
        
        # Print trainable parameters
        model.print_trainable_parameters()
        logger.info("LoRA adapters applied successfully")
        
        # Verify gradients are enabled
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Trainable parameters: {trainable_params:,}")
        
    else:
        # Enable gradient checkpointing to save memory
        model.gradient_checkpointing_enable()
    
    return model, tokenizer

def compute_metrics(eval_pred):
    """Compute evaluation metrics"""
    predictions, labels = eval_pred
    
    # Simple perplexity calculation
    predictions = torch.tensor(predictions)
    labels = torch.tensor(labels)
    
    # Mask padding tokens
    mask = labels != -100
    if mask.sum() == 0:
        return {"perplexity": float('inf')}
    
    # Calculate cross entropy loss
    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
    vocab_size = predictions.shape[-1]
    
    shift_predictions = predictions[..., :-1, :].contiguous().view(-1, vocab_size)
    shift_labels = labels[..., 1:].contiguous().view(-1)
    shift_mask = mask[..., 1:].contiguous().view(-1)
    
    losses = loss_fct(shift_predictions, shift_labels)
    masked_losses = losses * shift_mask
    
    avg_loss = masked_losses.sum() / shift_mask.sum()
    perplexity = torch.exp(avg_loss).item()
    
    return {"perplexity": perplexity}

def clear_memory():
    """Clear GPU memory"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def create_model_card(output_dir):
    """Create a comprehensive model card for the LoRA adapter"""
    
    model_card_content = """---
license: apache-2.0
base_model: Qwen/Qwen2.5-7B
library_name: peft
tags:
- text-to-speech
- ssml
- qwen2.5
- lora
- peft
language:
- en
- fr
pipeline_tag: text-generation
---

# Qwen2.5-7B SSML LoRA Adapter

This is a LoRA (Low-Rank Adaptation) fine-tuned version of Qwen2.5-7B for converting plain text to SSML (Speech Synthesis Markup Language) with appropriate pause predictions.

## Model Details

- **Base Model**: Qwen/Qwen2.5-7B
- **Fine-tuning Method**: LoRA (Low-Rank Adaptation)
- **Task**: Text-to-SSML conversion with pause prediction
- **Languages**: English, French (and others supported by base model)

## Usage

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Load base model and tokenizer
base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B")

# Load LoRA adapter
model = PeftModel.from_pretrained(base_model, "jonahdvt/qwen-ssml-lora")

# Prepare input
instruction = "Convert text to SSML with pauses:"
text = "Hello, how are you today? I hope everything is going well."
formatted_input = f"### Task:\\n{instruction}\\n\\n### Text:\\n{text}\\n\\n### SSML:\\n"

# Generate
inputs = tokenizer(formatted_input, return_tensors="pt").to(model.device)
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
ssml_output = response.split("### SSML:\\n")[-1]
print(ssml_output)
```

## Training Details

- **LoRA Rank**: 8
- **LoRA Alpha**: 16
- **Target Modules**: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- **Training Epochs**: 5
- **Batch Size**: 1 (with gradient accumulation)
- **Learning Rate**: 3e-4

## License

This model is released under the Apache 2.0 license, same as the base Qwen2.5-7B model.
"""

    readme_path = os.path.join(output_dir, "README.md")
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(model_card_content)
    
    logger.info(f"Model card created at {readme_path}")

def push_lora_to_hub(local_path, repo_id):
    """Manually push LoRA adapter to Hugging Face Hub"""
    
    try:
        from huggingface_hub import HfApi, create_repo
        
        # Get HF token
        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            logger.error("HF_TOKEN environment variable not set!")
            logger.info("Set your token with: export HF_TOKEN=your_token_here")
            return False
        
        # Initialize API
        api = HfApi(token=hf_token)
        
        # Create repository if it doesn't exist
        try:
            create_repo(repo_id, token=hf_token, exist_ok=True)
            logger.info(f"Repository {repo_id} created/verified")
        except Exception as e:
            logger.warning(f"Repository creation warning: {e}")
        
        # Upload all files in the output directory
        logger.info(f"Uploading files from {local_path} to {repo_id}...")
        
        # Upload the entire folder
        api.upload_folder(
            folder_path=local_path,
            repo_id=repo_id,
            token=hf_token,
            commit_message="Upload LoRA adapter for Qwen2.5-7B SSML conversion"
        )
        
        logger.info(f"✅ Successfully uploaded model to https://huggingface.co/{repo_id}")
        return True
        
    except ImportError:
        logger.error("huggingface_hub not installed. Install with: pip install huggingface_hub")
        return False
    except Exception as e:
        logger.error(f"Failed to upload to Hub: {e}")
        return False

def test_and_demonstrate_model(model_path, test_texts):
    """Test the model and create usage examples"""
    
    logger.info("Testing trained model and creating examples...")
    
    try:
        from peft import PeftModel
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        # Load models
        base_model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-7B",
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        model = PeftModel.from_pretrained(base_model, model_path)
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B")
        
        results = []
        
        for test_text in test_texts:
            instruction = "Convert text to SSML with pauses:"
            formatted_input = f"### Task:\n{instruction}\n\n### Text:\n{test_text}\n\n### SSML:\n"
            
            inputs = tokenizer(formatted_input, return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            ssml_output = response.split("### SSML:\n")[-1].strip()
            
            result = {
                "input": test_text,
                "output": ssml_output
            }
            results.append(result)
            
            logger.info(f"✅ Input: {test_text}")
            logger.info(f"✅ Output: {ssml_output}")
            logger.info("---")
        
        # Save examples
        examples_file = os.path.join(model_path, "examples.json")
        with open(examples_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        return results
        
    except Exception as e:
        logger.error(f"Model testing failed: {e}")
        return None

def main():
    """Main training function with memory optimizations"""
    logger.info("Starting optimized Qwen 7B SSML training...")
    
    # Set memory optimization environment variables
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    # Clear memory at start
    clear_memory()
    
    # Setup cache directories
    cache_dir = setup_cache_directories()
    
    # Setup output directory
    if os.environ.get('SCRATCH'):
        output_dir = os.path.join(os.environ['SCRATCH'], 'ssml_models')
    else:
        output_dir = './ssml_model_output'
    
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")
    
    # Load model and tokenizer with LoRA
    model, tokenizer = load_model_and_tokenizer(cache_dir=cache_dir, use_lora=True)
    
    # Load dataset with reduced max length
    data_path = "full_data"
    if not os.path.exists(data_path):
        logger.error(f"Data directory {data_path} not found!")
        return
    
    dataset = SSMLDataset(data_path, tokenizer, max_length=1024)  # Reduced from 2048
    
    # Split dataset
    total_size = len(dataset)
    train_size = int(0.9 * total_size)
    val_size = total_size - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Validation samples: {len(val_dataset)}")
    
    # Create data collator with minimal padding - THIS WAS MISSING!
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8,  # Small multiple for efficiency
        return_tensors="pt"
    )
    
    # Aggressive memory optimization training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=5,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=16,
        warmup_steps=50,
        logging_steps=20,
        eval_steps=10000,
        save_steps=10000,
        eval_strategy="steps",
        save_strategy="steps",
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        dataloader_pin_memory=False,
        dataloader_num_workers=0,
        bf16=True,
        gradient_checkpointing=True,
        optim="adamw_torch",
        learning_rate=3e-4,
        weight_decay=0.01,
        max_grad_norm=1.0,
        lr_scheduler_type="cosine",
        report_to=[],
        remove_unused_columns=False,
        group_by_length=False,
        load_best_model_at_end=False,
        save_total_limit=2,
        dataloader_drop_last=True,
        ignore_data_skip=True,
        prediction_loss_only=True,
        skip_memory_metrics=True,
        ddp_find_unused_parameters=False,
        # Hub configuration - better approach
        push_to_hub=False,  # We'll handle this manually for LoRA
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,  # Now properly defined!
    )
    
    # Start training
    logger.info("Starting training...")
    start_time = datetime.now()
    
    try:
        # Clear memory before training
        clear_memory()
        
        trainer.train()
        
        # Save the final model
        logger.info("Saving final model...")
        model.save_pretrained(output_dir)  # This saves LoRA adapters properly
        tokenizer.save_pretrained(output_dir)
        
        # Create model card and README
        create_model_card(output_dir)
        
        # Push to Hub manually for better control
        push_lora_to_hub(output_dir, "jonahdvt/qwen-ssml-lora")
        
        # Get training stats
        train_result = trainer.state.log_history
        
        end_time = datetime.now()
        training_duration = end_time - start_time
        
        # Save results
        results = {
            "model_name": "Qwen2.5-7B-SSML-LoRA",
            "training_duration": str(training_duration),
            "total_samples": total_size,
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset),
            "final_train_loss": train_result[-1].get("train_loss", "N/A") if train_result else "N/A",
            "final_eval_loss": train_result[-1].get("eval_loss", "N/A") if train_result else "N/A",
            "training_completed": True,
            "output_directory": output_dir,
            "timestamp": datetime.now().isoformat(),
            "using_lora": True,
            "max_sequence_length": 1024
        }
        
        results_file = os.path.join(os.environ.get('SCRATCH', '.'), 'result_stats.json')
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Training completed successfully in {training_duration}")
        logger.info(f"Results saved to: {results_file}")
        logger.info("Training and upload completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        # Save error info
        error_results = {
            "training_completed": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }
        
        results_file = os.path.join(os.environ.get('SCRATCH', '.'), 'result_stats.json')
        with open(results_file, 'w') as f:
            json.dump(error_results, f, indent=2)
        
        raise
    
    finally:
        # Cleanup
        del model
        del tokenizer
        clear_memory()

if __name__ == "__main__":
    # Check if required packages are available
    try:
        import peft
        logger.info("PEFT available")
    except ImportError as e:
        logger.error(f"Missing required package PEFT: {e}")
        logger.info("Install with: pip install peft")
    
    # Test texts for demonstration
    test_texts = [
        "Hello, how are you today? I hope everything is going well.",
        "Bonjour, comment allez-vous aujourd'hui? J'espère que tout va bien.",
        "The weather is beautiful today. Let's go for a walk in the park.",
        "This is a test sentence. It contains multiple parts, separated by punctuation."
    ]
    
    main()
    
    # Test the model after training
    output_dir = os.path.join(os.environ.get('SCRATCH', '.'), 'ssml_models')
    if os.path.exists(output_dir):
        test_and_demonstrate_model(output_dir, test_texts)(base)


