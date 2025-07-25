import json
import os
import random
import math
import re
import unicodedata
from transformers import BertTokenizerFast, BertForTokenClassification, Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset, Subset
from sklearn.metrics import classification_report
import numpy as np

# Configuration
MODEL_NAME = 'bert-base-multilingual-uncased'
MAX_LENGTH = 128
BATCH_SIZE = 64
NUM_EPOCHS = 10
LABEL_ALL_SUBTOKENS = False  # Only label first subtoken
RANDOM_SEED = 1  # <-- fixed seed for reproducibility
BOOTSTRAPPING = True  # Enable bootstrapping for robust performance estimation
NUM_BOOTSTRAP_RUNS = 10  # Number of bootstrap iterations

def normalize_text(text: str) -> str:
    """
    Apply Unicode NFC normalization and collapse multiple whitespace.
    """
    # 1. Unicode NFC normalization, and convert to lowercase
    text = unicodedata.normalize('NFC', text).lower()
    # 2. Collapse any whitespace (tabs/newlines/multiple spaces) into a single space
    text = re.sub(r'\s+', ' ', text).strip()
    return text

class PhraseBreakDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=MAX_LENGTH):
        self.examples = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        with open(data_path, 'r', encoding='utf-8') as f:
            raw = json.load(f)
        
        for key, entry in raw.items():
            raw_text = entry['x']  # Full text input
            # Apply normalization here
            # text = normalize_text(raw_text)
            text = raw_text  # If you want to keep original text, comment out normalization
            
            parsed = entry['y']['parsed_sequence']
            # Reconstruct sequence of words and break labels
            words = []
            labels = []  # 1 if break after word, 0 otherwise
            for seg in parsed:
                if seg['type'] == 'text':
                    # split text segment into words
                    for w in seg['text'].split():
                        # You might also choose to normalize each word again if needed:
                        # w = normalize_text(w)
                        words.append(w)
                        labels.append(0)  # default no break; may adjust below
                elif seg['type'] == 'break':
                    # assign break label to previous word if exists
                    if len(labels) > 0:
                        labels[-1] = 1

            # Tokenize and align labels
            encoding = tokenizer(
                words,
                is_split_into_words=True,
                return_offsets_mapping=False,
                truncation=True,
                padding='max_length',
                max_length=self.max_length
            )
            word_ids = encoding.word_ids()
            aligned_labels = []
            previous_word_idx = None
            for idx, word_idx in enumerate(word_ids):
                if word_idx is None:
                    aligned_labels.append(-100)
                elif word_idx != previous_word_idx:
                    # first subtoken of this word
                    aligned_labels.append(labels[word_idx])
                else:
                    # subsequent subtoken
                    if LABEL_ALL_SUBTOKENS:
                        aligned_labels.append(labels[word_idx])
                    else:
                        aligned_labels.append(-100)
                previous_word_idx = word_idx

            encoding['labels'] = aligned_labels
            self.examples.append({k: torch.tensor(v) for k, v in encoding.items()})

    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]


def create_bootstrap_dataset(original_indices, bootstrap_seed):
    """
    Create a bootstrap sample by resampling with replacement.
    """
    random.seed(bootstrap_seed)
    np.random.seed(bootstrap_seed)
    
    # Resample with replacement to create bootstrap dataset of same size
    bootstrap_indices = np.random.choice(
        original_indices, 
        size=len(original_indices), 
        replace=True
    ).tolist()
    
    return bootstrap_indices


def train_single_model(dataset, train_indices, eval_indices, tokenizer, run_id=0):
    """
    Train a single model instance and return performance metrics.
    """
    # Create datasets
    train_dataset = Subset(dataset, train_indices)
    eval_dataset = Subset(dataset, eval_indices)
    
    # Initialize fresh model for each bootstrap run
    model = BertForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,  # break vs no break
        id2label={0: 'NO_BREAK', 1: 'BREAK'},
        label2id={'NO_BREAK': 0, 'BREAK': 1}
    )
    
    # Training arguments with unique output directory
    training_args = TrainingArguments(
        output_dir=f'./results_bootstrap_{run_id}',
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        eval_strategy='epoch',
        save_strategy='epoch',
        save_total_limit=1,
        logging_dir=f'./logs_bootstrap_{run_id}',
        logging_steps=50,
        report_to=[]
    )
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )
    
    # Train
    trainer.train()
    
    # Evaluate and get metrics
    predictions, labels, metrics = trainer.predict(eval_dataset)
    eval_loss = metrics.get('eval_loss', metrics.get('test_loss'))
    perplexity = math.exp(eval_loss)
    
    # Calculate F1 scores
    preds_flat = torch.argmax(torch.tensor(predictions), dim=2).view(-1).numpy()
    labels_flat = torch.tensor(labels).view(-1).numpy()
    valid_indices = labels_flat != -100
    preds_filtered = preds_flat[valid_indices]
    labels_filtered = labels_flat[valid_indices]
    
    report_dict = classification_report(
        labels_filtered,
        preds_filtered,
        target_names=['NO_BREAK', 'BREAK'],
        output_dict=True
    )
    
    f1_no_break = report_dict['NO_BREAK']['f1-score']
    f1_break = report_dict['BREAK']['f1-score']
    
    return {
        'perplexity': perplexity,
        'f1_no_break': f1_no_break,
        'f1_break': f1_break,
        'eval_loss': eval_loss,
        'model': model
    }


def main():
    # Paths
    data_path = os.path.join(
        os.getcwd(),
        '/home/mila/d/dauvetj/mon_projet_TTS/Code/ssml_models/jonah/bdd.json'
    )
    
    # Initialize tokenizer
    tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)
    
    # Prepare dataset
    dataset = PhraseBreakDataset(data_path, tokenizer)
    
    # Create initial train/eval split
    random.seed(RANDOM_SEED)
    all_indices = list(range(len(dataset)))
    random.shuffle(all_indices)
    split_idx = int(0.9 * len(all_indices))
    train_indices = all_indices[:split_idx]
    eval_indices = all_indices[split_idx:]
    
    if BOOTSTRAPPING:
        print(f"Running bootstrapping with {NUM_BOOTSTRAP_RUNS} iterations...")
        bootstrap_results = []
        
        for i in range(NUM_BOOTSTRAP_RUNS):
            print(f"\nBootstrap run {i+1}/{NUM_BOOTSTRAP_RUNS}")
            
            # Create bootstrap sample of training data
            bootstrap_train_indices = create_bootstrap_dataset(
                train_indices, 
                bootstrap_seed=RANDOM_SEED + i
            )
            
            # Train model on bootstrap sample
            results = train_single_model(
                dataset, 
                bootstrap_train_indices, 
                eval_indices, 
                tokenizer, 
                run_id=i
            )
            
            bootstrap_results.append(results)
            
            print(f"Run {i+1} - Perplexity: {results['perplexity']:.4f}, "
                  f"F1 NO_BREAK: {results['f1_no_break']:.4f}, "
                  f"F1 BREAK: {results['f1_break']:.4f}")
        
        # Calculate statistics across bootstrap runs
        perplexities = [r['perplexity'] for r in bootstrap_results]
        f1_no_breaks = [r['f1_no_break'] for r in bootstrap_results]
        f1_breaks = [r['f1_break'] for r in bootstrap_results]
        
        print("\n" + "="*60)
        print("BOOTSTRAP RESULTS SUMMARY")
        print("="*60)
        print(f"Perplexity - Mean: {np.mean(perplexities):.4f} ± {np.std(perplexities):.4f}")
        print(f"             Range: [{np.min(perplexities):.4f}, {np.max(perplexities):.4f}]")
        print(f"F1 NO_BREAK - Mean: {np.mean(f1_no_breaks):.4f} ± {np.std(f1_no_breaks):.4f}")
        print(f"              Range: [{np.min(f1_no_breaks):.4f}, {np.max(f1_no_breaks):.4f}]")
        print(f"F1 BREAK - Mean: {np.mean(f1_breaks):.4f} ± {np.std(f1_breaks):.4f}")
        print(f"           Range: [{np.min(f1_breaks):.4f}, {np.max(f1_breaks):.4f}]")
        
        # Save the best performing model
        best_idx = np.argmax(f1_breaks)  # or choose based on other criteria
        best_model = bootstrap_results[best_idx]['model']
        best_model.save_pretrained('./bert_phrase_break_model_best')
        tokenizer.save_pretrained('./bert_phrase_break_model_best')
        print(f"\nBest model (run {best_idx+1}) saved to './bert_phrase_break_model_best'")
        
    else:
        # Original single training run
        print("Running single training run (no bootstrapping)...")
        results = train_single_model(dataset, train_indices, eval_indices, tokenizer)
        
        print(f"Perplexity: {results['perplexity']:.4f}")
        print(f"F1 (NO_BREAK, class 0): {results['f1_no_break']:.4f}")
        print(f"F1 (BREAK,    class 1): {results['f1_break']:.4f}")
        
        # Save model and tokenizer
        results['model'].save_pretrained('./bert_phrase_break_model')
        tokenizer.save_pretrained('./bert_phrase_break_model')


if __name__ == '__main__':
    main()