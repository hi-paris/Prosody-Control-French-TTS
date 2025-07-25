import os
import json
from datasets import load_dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig # Import for 4-bit quantization
)
import torch
import logging
from datetime import datetime
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training # LoRA imports

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_scratch_directory():
    """Setup output directory to use SCRATCH if available."""
    output_dir = None
    if os.environ.get('SCRATCH'):
        output_dir = os.path.join(os.environ['SCRATCH'], 'qwen_finetuned_z2y_lora') # Changed output dir name
        logger.info(f"Using SCRATCH for output directory: {output_dir}")
    else:
        output_dir = "./qwen_finetuned_z2y_lora"
        logger.warning(f"SCRATCH environment variable not found. Using default output directory: {output_dir}")
    
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def push_to_hub_manual(local_path, repo_id):
    """Manually pushes the model (LoRA adapter) to Hugging Face Hub."""
    try:
        from huggingface_hub import HfApi, create_repo
        
        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            logger.error("HF_TOKEN environment variable not set! Please set your Hugging Face token.")
            logger.info("You can set it with: export HF_TOKEN=your_token_here")
            return False
        
        api = HfApi(token=hf_token)
        
        try:
            create_repo(repo_id, token=hf_token, exist_ok=True, private=False) # Ensure public if desired
            logger.info(f"Repository {repo_id} created/verified.")
        except Exception as e:
            logger.warning(f"Repository creation warning: {e}")
        
        logger.info(f"Uploading LoRA adapter files from {local_path} to {repo_id}...")
        
        api.upload_folder(
            folder_path=local_path,
            repo_id=repo_id,
            token=hf_token,
            commit_message="Upload fine-tuned Qwen2.5-7B LoRA adapter (z to y conversion)"
        )
        
        logger.info(f"✅ Successfully uploaded LoRA adapter to https://huggingface.co/{repo_id}")
        return True
        
    except ImportError:
        logger.error("huggingface_hub not installed. Please install with: pip install huggingface_hub")
        return False
    except Exception as e:
        logger.error(f"Failed to upload to Hugging Face Hub: {e}")
        return False

def main():
    logger.info("Starting Qwen Z to Y fine-tuning with LoRA...")

    # Set memory optimization environment variables
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # Paths
    data_dir = "/home/mila/d/dauvetj/mon_projet_TTS/Code/ssml_models/jonah/full_data_xyz"
    data_files = {
        "train": os.path.join(data_dir, "train.json"),
        "validation": os.path.join(data_dir, "val.json"),
        "test": os.path.join(data_dir, "test.json")
    }

    # Load dataset
    raw_datasets = load_dataset("json", data_files=data_files)
    logger.info("Dataset loaded.")

    # Model & Tokenizer
    model_name = "Qwen/Qwen2.5-7B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Add a pad token if not already present (crucial for batching)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
        logger.info(f"Added pad token: {tokenizer.pad_token}")

    # BitsAndBytes configuration for 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4", # Use NF4 quantization
        bnb_4bit_compute_dtype=torch.bfloat16, # Compute in bfloat16
        bnb_4bit_use_double_quant=True, # Double quantization for slightly better performance
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config, # Apply 4-bit quantization
        device_map="auto",
        torch_dtype=torch.bfloat16 # Still specify bfloat16 for non-quantized parts and general compute
    )

    # Enable gradient checkpointing and prepare model for k-bit training (LoRA)
    model.gradient_checkpointing_enable() 
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True) # Essential for LoRA with quantization

    # LoRA configuration - targeting the right modules for Qwen2.5
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,  # LoRA rank
        lora_alpha=16, # LoRA alpha (scaling factor)
        lora_dropout=0.1,
        target_modules=[
            "q_proj",
            "k_proj", 
            "v_proj",
            "o_proj",
            "gate_proj",  # Specific to Qwen's MoE-like architecture
            "up_proj",    # Specific to Qwen's MoE-like architecture
            "down_proj"   # Specific to Qwen's MoE-like architecture
        ],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameters to verify LoRA is working
    model.print_trainable_parameters()
    logger.info("LoRA adapters applied successfully to the quantized model.")
    logger.info(f"Model {model_name} and tokenizer loaded.")

    # Preprocessing: use 'z' as input, 'y' as target
    def tokenize_function(example):
        input_str = example["z"].strip()
        target_str = example["y"].strip()
        
        # Concatenate for causal LM: Input + Target
        # The labels will be -100 for input tokens, and actual token IDs for target tokens
        # Set a reasonable max_length to avoid extremely long sequences, which are memory intensive
        # Adjusted max_seq_length to 768, which is a good balance for Qwen and common tasks.
        max_seq_length = 768 
        
        # The prompt template structure is crucial for instruction-tuned models like Qwen.
        # Use a simplified instruction format suitable for z -> y.
        # Ensure the prompt doesn't interfere with label masking.
        # Format as a simple chat turn, with model expected to respond.
        prompt = f"### Instruction:\nConvert text Z to text Y.\n\n### Input Z:\n{input_str}\n\n### Output Y:\n"
        
        full_text = prompt + target_str + tokenizer.eos_token # Add EOS token at the very end

        # Tokenize the full sequence
        tokenized_output = tokenizer(
            full_text,
            truncation=True,
            max_length=max_seq_length,
            padding=False,
            return_tensors="pt" # Return PyTorch tensors
        )
        
        input_ids = tokenized_output["input_ids"].squeeze() # Remove batch dimension
        attention_mask = tokenized_output["attention_mask"].squeeze()

        labels = input_ids.clone() # Labels are the same as input_ids for causal LM

        # Mask the prompt part so we only compute loss on the target_str (Output Y)
        # Find the length of the prompt up to "### Output Y:\n"
        # We need to tokenize the prompt part separately to get its exact token length without special tokens potentially
        prompt_tokens_len = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
        
        # Set labels for the prompt part to -100
        labels[:prompt_tokens_len] = -100
        
        return {
            "input_ids": input_ids.tolist(), # Convert to list for dataset
            "attention_mask": attention_mask.tolist(),
            "labels": labels.tolist()
        }

    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=False, # Process one example at a time
        remove_columns=list(raw_datasets["train"].features.keys()), # Remove original text columns
        load_from_cache_file=False # Set to False to ensure changes to tokenize_function are applied
    )
    logger.info("Dataset tokenized.")
    logger.info(f"Sample tokenized input_ids: {tokenized_datasets['train'][0]['input_ids']}")
    logger.info(f"Sample tokenized labels: {tokenized_datasets['train'][0]['labels']}")


    # Data collator for padding
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    logger.info("Data collator initialized.")

    # Setup output directory using SCRATCH
    output_dir = setup_scratch_directory()
    repo_id = "jonahdvt/qwen-z2y-lora" # Updated Hugging Face Hub repository ID for LoRA

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1, # Very small batch size to prevent OOM
        per_device_eval_batch_size=1,  # Same for evaluation
        gradient_accumulation_steps=32, # Increased accumulation to compensate for batch_size=1
        eval_strategy="steps",
        eval_steps=5000, # Evaluate every 500 steps
        logging_steps=50, # Log every 50 steps
        save_steps=5000, # Save checkpoint every 500 steps
        num_train_epochs=5, # Number of training epochs
        learning_rate=3e-4, # Common learning rate for LoRA
        weight_decay=0.01,
        warmup_steps=100,
        fp16=False, # Use bf16 if supported, otherwise rely on 4-bit quantization
        bf16=True, # Explicitly use bfloat16 for compute if your GPU supports it
        push_to_hub=False, # We will manually push after saving the model
        report_to="none", # Disable external reporting
        save_total_limit=2, # Keep only the last 2 checkpoints
        load_best_model_at_end=True, # Load the best model based on eval_loss at the end of training
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        remove_unused_columns=False, # Important to keep 'labels' column
        gradient_checkpointing=True, # Ensure gradient checkpointing is enabled
        dataloader_num_workers=0, # Set to 0 to simplify debugging OOM
        dataloader_pin_memory=True, # Can be True or False, depends on system, True is usually faster
    )
    logger.info("Training arguments set up.")

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
    )
    logger.info("Trainer initialized.")

    # Train
    logger.info("Starting training...")
    start_time = datetime.now()
    try:
        trainer.train()
        end_time = datetime.now()
        training_duration = end_time - start_time
        logger.info(f"Training completed in {training_duration}.")

        # Save the final LoRA adapter weights
        logger.info(f"Saving final LoRA adapter to {output_dir}...")
        trainer.model.save_pretrained(output_dir) # Save PEFT model
        tokenizer.save_pretrained(output_dir) # Save tokenizer with the model
        logger.info("LoRA adapter and tokenizer saved locally.")

        # Manually push to Hugging Face Hub
        logger.info(f"Attempting to push LoRA adapter to Hugging Face Hub: {repo_id}")
        push_to_hub_manual(output_dir, repo_id)

        logger.info("Training and model push process finished.")
    except Exception as e:
        logger.error(f"An error occurred during training: {e}")
        logger.error("Training failed. Please review the error message and adjust parameters.")
        # Re-raise the exception to show the full traceback for debugging
        raise

if __name__ == "__main__":
    main()
