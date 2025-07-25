import whisper_timestamped as whisper
import os

def create_word_level_labels():
    """Create individual word labels for Audacity manual alignment"""
    
    # File paths
    audio_file = "/home/mila/d/dauvetj/mon_projet_TTS/Data/gold/Aznavour_1.wav"
    labels_file = "/home/mila/d/dauvetj/mon_projet_TTS/Data/gold/labels_az1.txt"
    output_file = "/home/mila/d/dauvetj/mon_projet_TTS/Data/word_labels_for_audacity.txt"
    
    # Load Whisper model
    print("Loading Whisper model...")
    model = whisper.load_model("large-v3")
    
    # Transcribe with word-level timestamps
    print(f"Transcribing audio: {audio_file}")
    result = whisper.transcribe(model, audio_file, language="fr")
    
    # Read your boundary timestamps
    boundaries = []
    with open(labels_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                timestamp = float(parts[0])
                label = parts[2]
                boundaries.append((timestamp, label))
    
    boundaries.sort(key=lambda x: x[0])
    
    # Create Audacity labels
    audacity_labels = []
    
    # Add your original boundaries first
    for i, (timestamp, label) in enumerate(boundaries):
        end_time = boundaries[i + 1][0] if i < len(boundaries) - 1 else result['segments'][-1]['end']
        audacity_labels.append((timestamp, end_time, f"[{label}]"))
    
    # Add individual words within each segment
    for segment in result['segments']:
        for word_info in segment.get('words', []):
            word_start = word_info['start']
            word_end = word_info['end']
            word_text = word_info['text'].strip()
            
            # Find which segment this word belongs to
            segment_label = "unknown"
            for i, (boundary_time, boundary_label) in enumerate(boundaries):
                next_boundary = boundaries[i + 1][0] if i < len(boundaries) - 1 else float('inf')
                if boundary_time <= word_start < next_boundary:
                    segment_label = boundary_label
                    break
            
            # Only add words from numbered segments (not breaks)
            if segment_label.isdigit():
                audacity_labels.append((word_start, word_end, word_text))
    
    # Sort all labels by start time
    audacity_labels.sort(key=lambda x: x[0])
    
    # Write Audacity label file
    with open(output_file, 'w', encoding='utf-8') as f:
        for start, end, text in audacity_labels:
            f.write(f"{start:.6f}\t{end:.6f}\t{text}\n")
    
    print(f"Word-level labels saved to: {output_file}")
    print(f"Created {len(audacity_labels)} labels")
    
    # Also create a simplified version with just segment boundaries
    simple_output = "Data/segment_boundaries_for_audacity.txt"
    with open(simple_output, 'w', encoding='utf-8') as f:
        for timestamp, label in boundaries:
            f.write(f"{timestamp:.6f}\t{timestamp:.6f}\t{label}\n")
    
    print(f"Segment boundaries saved to: {simple_output}")

if __name__ == "__main__":
    create_word_level_labels()