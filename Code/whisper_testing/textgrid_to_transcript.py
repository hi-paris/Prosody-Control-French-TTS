#!/usr/bin/env python3
"""
TextGrid to Gold Transcript Converter
Extracts text from a Praat TextGrid file to create a simple gold transcript with spelling correction.
"""

import re
import sys
from pathlib import Path
import spacy
from spacy.lang.fr import French

def parse_textgrid(file_path):
    """
    Parse a TextGrid file and extract text intervals.
    
    Args:
        file_path (str): Path to the TextGrid file
        
    Returns:
        list: List of tuples containing (start_time, end_time, text)
    """
    intervals = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Find all interval blocks
        interval_pattern = r'intervals \[(\d+)\]:\s*xmin = ([\d.]+)\s*xmax = ([\d.]+)\s*text = "([^"]*)"'
        matches = re.findall(interval_pattern, content, re.MULTILINE | re.DOTALL)
        
        for match in matches:
            interval_num, xmin, xmax, text = match
            start_time = float(xmin)
            end_time = float(xmax)
            
            # Skip empty text intervals
            if text.strip():
                intervals.append((start_time, end_time, text.strip().lower()))
    
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return []
    except Exception as e:
        print(f"Error reading file: {e}")
        return []
    
    # Sort intervals by start time
    intervals.sort(key=lambda x: x[0])
    return intervals

def correct_spelling(text):
    """
    Correct spelling mistakes using spaCy French model.
    
    Args:
        text (str): Input text to correct
        
    Returns:
        str: Corrected text
    """
    try:
        # Load French language model
        nlp = spacy.load("fr_core_news_sm")
        
        # Common French spelling corrections (manual dictionary for common ASR errors)
        corrections = {
            "predestinait": "prédestinait",
            "appatride": "apatride",
            "magellane": "Magellan",
            "qu'est-ce": "Qu'est-ce"
        }
        
        # Apply manual corrections first
        corrected_text = text
        for wrong, correct in corrections.items():
            corrected_text = re.sub(r'\b' + re.escape(wrong) + r'\b', correct, corrected_text, flags=re.IGNORECASE)
        
        # Process with spaCy for additional analysis
        doc = nlp(corrected_text)
        
        # Additional processing could be added here for more sophisticated corrections
        # For now, we rely on the manual corrections dictionary
        
        return corrected_text
        
    except OSError:
        print("Warning: French spaCy model not found. Install with: python -m spacy download fr_core_news_sm")
        return text
    except Exception as e:
        print(f"Warning: Spelling correction failed: {e}")
        return text

def create_gold_transcript(intervals, output_file=None):
    """
    Create a gold transcript from text intervals with spelling correction.
    
    Args:
        intervals (list): List of (start_time, end_time, text) tuples
        output_file (str, optional): Output file path. If None, prints to stdout.
    """
    if not intervals:
        return
    
    # Extract just the text and join with spaces
    transcript_text = ' '.join([text for _, _, text in intervals])
    
    # Correct spelling mistakes
    corrected_text = correct_spelling(transcript_text)
    
    if output_file:
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(corrected_text)
        except Exception as e:
            print(f"Error writing output file: {e}")
    else:
        print(corrected_text)

def main():
    """Main function to process TextGrid file."""
    
    # Default input file path from your example
    default_input = "/home/mila/d/dauvetj/mon_projet_TTS/Data/gold/word_level_az1.TextGrid"
    
    # Check command line arguments
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    else:
        input_file = default_input
    
    # Parse the TextGrid file
    intervals = parse_textgrid(input_file)
    
    # Determine output file
    output_file = "/home/mila/d/dauvetj/mon_projet_TTS/Data/gold/az1_transcript.txt"
    
    # Create the gold transcript
    create_gold_transcript(intervals, output_file)

if __name__ == "__main__":
    main()

