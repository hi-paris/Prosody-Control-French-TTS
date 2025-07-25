import numpy as np
import pandas as pd
from pathlib import Path
import re
import time


def parse_textgrid(textgrid_path):
    """Parse TextGrid file to extract word intervals."""
    intervals = []
    with open(textgrid_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    pattern = r'intervals \[(\d+)\]:\s*xmin = ([\d.]+)\s*xmax = ([\d.]+)\s*text = "([^"]*)"'
    for match in re.findall(pattern, content):
        _, xmin, xmax, text = match
        if text.strip():
            intervals.append({
                'start': float(xmin),
                'end': float(xmax),
                'text': text.strip(),
                'duration': float(xmax) - float(xmin)
            })
    return intervals

# extract_whisper_words function is removed as it's specific to Whisper processing

def evaluate_nemo_textgrid(nemo_textgrid_path, gold_intervals, audio_duration):
    """Evaluate NeMo TextGrid against gold standard."""
    print(f"Evaluating NeMo TextGrid: {nemo_textgrid_path}")
    
    # Parse NeMo TextGrid
    nemo_intervals = parse_textgrid(nemo_textgrid_path)
    print(f"NeMo intervals: {len(nemo_intervals)}")
    
    if not nemo_intervals:
        print("No intervals found in NeMo TextGrid!")
        return None
    
    # Create a mock whisper result for multilevel stats (since NeMo doesn't provide segments)
    # We'll create artificial segments based on silence gaps or fixed windows
    mock_whisper_result = create_mock_segments_from_intervals(nemo_intervals, audio_duration)
    
    # Calculate all metrics
    stats = calculate_multilevel_stats(gold_intervals, nemo_intervals, mock_whisper_result, audio_duration)
    stats['model'] = 'NeMo'
    stats['processing_time'] = 0  # Not measured for pre-generated TextGrid
    
    return stats

def create_mock_segments_from_intervals(intervals, audio_duration, max_segment_duration=30):
    """Create mock Whisper-style segments from intervals for sentence-level analysis."""
    if not intervals:
        return {'segments': []}
    
    segments = []
    if not intervals: # Should be caught by the above, but good for robustness
        return {'segments': segments}

    current_segment_start = intervals[0]['start']
    current_segment_end = intervals[0]['end']
    
    for i, interval in enumerate(intervals[1:], 1):
        # If there's a gap > 1 second or segment is getting too long, create new segment
        gap = interval['start'] - current_segment_end
        segment_duration = current_segment_end - current_segment_start
        
        if gap > 1.0 or segment_duration > max_segment_duration:
            # Close current segment
            segments.append({
                'start': current_segment_start,
                'end': current_segment_end
            })
            # Start new segment
            current_segment_start = interval['start']
            current_segment_end = interval['end']
        else:
            # Extend current segment
            current_segment_end = interval['end']
    
    # Add final segment
    if current_segment_start < audio_duration: # Ensure segment has some duration within audio
         segments.append({
            'start': current_segment_start,
            'end': current_segment_end
        })
    
    return {'segments': segments}

def normalize_text(text):
    """Normalize text for alignment."""
    return re.sub(r'[^\w\s]', '', text.lower()).strip()

def align_intervals(gold_intervals, pred_intervals):
    """Align gold and predicted intervals based on text similarity."""
    aligned_pairs = []
    used_pred = set()
    
    for gold in gold_intervals:
        gold_norm = normalize_text(gold['text'])
        best_match = None
        best_score = 0
        
        for i, pred in enumerate(pred_intervals):
            if i in used_pred:
                continue
                
            pred_norm = normalize_text(pred['text'])
            
            # Simple similarity scoring
            if gold_norm == pred_norm:
                score = 1.0
            elif gold_norm in pred_norm or pred_norm in gold_norm:
                score = 0.8
            elif any(word in pred_norm.split() for word in gold_norm.split()):
                score = 0.5
            else:
                score = 0
            
            if score > best_score and score > 0.4: # Threshold for considering a match
                best_score = score
                best_match = (i, pred)
        
        if best_match:
            used_pred.add(best_match[0])
            aligned_pairs.append((gold, best_match[1]))
    
    return aligned_pairs

def calculate_metrics(aligned_pairs, total_gold):
    """Calculate all metrics from aligned pairs."""
    if not aligned_pairs:
        return {
            'ARR': 0.0, 'MAE_start': float('inf'), 'MAE_end': float('inf'), 
            'MAE_duration': float('inf'), 'RMSE_start': float('inf'), 
            'RMSE_end': float('inf'), 'RMSE_duration': float('inf'),
            'count': 0
        }
    
    start_errors = [abs(gold['start'] - pred['start']) for gold, pred in aligned_pairs]
    end_errors = [abs(gold['end'] - pred['end']) for gold, pred in aligned_pairs]
    duration_errors = [abs(gold['duration'] - pred['duration']) for gold, pred in aligned_pairs]
    
    return {
        'ARR': len(aligned_pairs) / total_gold if total_gold > 0 else 0,
        'MAE_start': np.mean(start_errors),
        'MAE_end': np.mean(end_errors),
        'MAE_duration': np.mean(duration_errors),
        'RMSE_start': np.sqrt(np.mean(np.array(start_errors)**2)),
        'RMSE_end': np.sqrt(np.mean(np.array(end_errors)**2)),
        'RMSE_duration': np.sqrt(np.mean(np.array(duration_errors)**2)),
        'count': len(aligned_pairs)
    }

def get_intervals_in_window(intervals, start_time, end_time):
    """Get intervals that overlap with a time window."""
    return [
        interval for interval in intervals
        if interval['start'] < end_time and interval['end'] > start_time
    ]

def get_intervals_in_segment(intervals, segment):
    """Get intervals that overlap with a Whisper segment."""
    seg_start = segment.get('start', 0)
    # Ensure seg_end is valid, especially if segment is just {'start': x, 'end': y}
    seg_end = segment.get('end', seg_start + 1 if seg_start is not None else 1) 
    if seg_start is None or seg_end is None : return [] # Should not happen with create_mock_segments
    return get_intervals_in_window(intervals, seg_start, seg_end)


def calculate_multilevel_stats(gold_intervals, pred_intervals, model_output_segments, audio_duration):
    """Calculate statistics at all levels.
       'model_output_segments' is whisper_result for Whisper, or mock_whisper_result for NeMo.
    """
    results = {}
    
    # 1. ENTIRE AUDIO
    aligned_pairs_entire = align_intervals(gold_intervals, pred_intervals)
    results['entire_audio'] = calculate_metrics(aligned_pairs_entire, len(gold_intervals))
    
    # 2. 15-SECOND WINDOWS
    window_stats_list = []
    for start_time_window in range(0, int(audio_duration), 15):
        end_time_window = min(start_time_window + 15, audio_duration)
        gold_window = get_intervals_in_window(gold_intervals, start_time_window, end_time_window)
        pred_window = get_intervals_in_window(pred_intervals, start_time_window, end_time_window)
        
        if gold_window: # Only calculate if there are gold intervals in the window
            aligned_window = align_intervals(gold_window, pred_window)
            stats_window = calculate_metrics(aligned_window, len(gold_window))
            stats_window['window_start'] = start_time_window
            window_stats_list.append(stats_window)
    
    if window_stats_list:
        results['window_15s'] = {
            'mean_ARR': np.mean([w['ARR'] for w in window_stats_list]),
            'mean_MAE_start': np.mean([w['MAE_start'] for w in window_stats_list if w['MAE_start'] != float('inf')]),
            'mean_MAE_end': np.mean([w['MAE_end'] for w in window_stats_list if w['MAE_end'] != float('inf')]),
            'mean_MAE_duration': np.mean([w['MAE_duration'] for w in window_stats_list if w['MAE_duration'] != float('inf')]),
            'std_ARR': np.std([w['ARR'] for w in window_stats_list]),
            'window_count': len(window_stats_list)
        }
    else:
        results['window_15s'] = {'mean_ARR': 0, 'window_count': 0, 'mean_MAE_start': float('inf'), 'mean_MAE_end': float('inf'), 'mean_MAE_duration': float('inf')}
    
    # 3. SENTENCE LEVEL (Using segments from model_output_segments)
    sentence_stats_list = []
    if 'segments' in model_output_segments and model_output_segments['segments']:
        for segment in model_output_segments['segments']:
            gold_sentence = get_intervals_in_segment(gold_intervals, segment)
            pred_sentence = get_intervals_in_segment(pred_intervals, segment)
            
            if gold_sentence: # Only calculate if there are gold intervals in the segment
                aligned_sentence = align_intervals(gold_sentence, pred_sentence)
                stats_sentence = calculate_metrics(aligned_sentence, len(gold_sentence))
                sentence_stats_list.append(stats_sentence)
    
    if sentence_stats_list:
        results['sentence'] = {
            'mean_ARR': np.mean([s['ARR'] for s in sentence_stats_list]),
            'mean_MAE_start': np.mean([s['MAE_start'] for s in sentence_stats_list if s['MAE_start'] != float('inf')]),
            'mean_MAE_end': np.mean([s['MAE_end'] for s in sentence_stats_list if s['MAE_end'] != float('inf')]),
            'mean_MAE_duration': np.mean([s['MAE_duration'] for s in sentence_stats_list if s['MAE_duration'] != float('inf')]),
            'sentence_count': len(sentence_stats_list)
        }
    else:
        results['sentence'] = {'mean_ARR': 0, 'sentence_count': 0, 'mean_MAE_start': float('inf'), 'mean_MAE_end': float('inf'), 'mean_MAE_duration': float('inf')}
    
    # 4. WORD LEVEL (individual word statistics from entire audio alignment)
    if aligned_pairs_entire: # Use the alignment from the entire audio
        word_start_errors = [abs(gold['start'] - pred['start']) for gold, pred in aligned_pairs_entire]
        word_end_errors = [abs(gold['end'] - pred['end']) for gold, pred in aligned_pairs_entire]
        word_duration_errors = [abs(gold['duration'] - pred['duration']) for gold, pred in aligned_pairs_entire]
        
        results['word'] = {
            'mean_start_error': np.mean(word_start_errors) if word_start_errors else float('inf'),
            'mean_end_error': np.mean(word_end_errors) if word_end_errors else float('inf'),
            'mean_duration_error': np.mean(word_duration_errors) if word_duration_errors else float('inf'),
            'median_start_error': np.median(word_start_errors) if word_start_errors else float('inf'),
            'median_end_error': np.median(word_end_errors) if word_end_errors else float('inf'),
            'median_duration_error': np.median(word_duration_errors) if word_duration_errors else float('inf'),
            'max_start_error': np.max(word_start_errors) if word_start_errors else float('inf'),
            'max_end_error': np.max(word_end_errors) if word_end_errors else float('inf'),
            'word_count': len(aligned_pairs_entire)
        }
    else:
        results['word'] = {'word_count': 0, 'mean_start_error': float('inf'), 'mean_end_error': float('inf'), 
                           'mean_duration_error': float('inf'), 'median_start_error': float('inf'),
                           'median_end_error': float('inf'), 'median_duration_error': float('inf'),
                           'max_start_error': float('inf'), 'max_end_error': float('inf')}
                           
    return results

def run_nemo_evaluation(): # Renamed for clarity
    """Main NeMo evaluation function."""
    # Paths
    audio_path = "/home/mila/d/dauvetj/mon_projet_TTS/Data/gold/Aznavour_1.wav"
    gold_path = "/home/mila/d/dauvetj/mon_projet_TTS/Data/gold/word_level_az1.TextGrid"
    nemo_path = "/home/mila/d/dauvetj/mon_projet_TTS/Data/nemo_textgrids/Aznavour_1.TextGrid"
    output_dir = Path("/home/mila/d/dauvetj/mon_projet_TTS/Data/nemo_evaluation_results") # Adjusted output dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load gold standard
    print("Loading gold standard...")
    gold_intervals = parse_textgrid(gold_path)
    print(f"Gold intervals: {len(gold_intervals)}")
    
    if not gold_intervals:
        print("No gold intervals found. Exiting.")
        return

    # Estimate audio duration
    audio_duration = 120 # fallback
    try:
        import librosa
        # audio, sr = librosa.load(audio_path, sr=None) # Load with original sr
        # audio_duration = librosa.get_duration(y=audio, sr=sr)
        # For simplicity if only duration is needed and sr is known/assumed for whisper
        audio_duration = librosa.get_duration(path=audio_path)
        print(f"Audio duration: {audio_duration:.2f}s (from librosa)")
    except Exception as e:
        print(f"Could not load audio duration with librosa: {e}. Using fallback: {audio_duration}s")
    
    all_results = []
    
    # Test NeMo
    print("\n--- Testing NeMo ---")
    if Path(nemo_path).exists():
        try:
            nemo_stats = evaluate_nemo_textgrid(nemo_path, gold_intervals, audio_duration)
            if nemo_stats:
                all_results.append(nemo_stats)
                # Print summary
                ea = nemo_stats['entire_audio']
                print(f"  NeMo Results - ARR: {ea['ARR']:.3f}, MAE_start: {ea['MAE_start']:.3f}s, MAE_duration: {ea['MAE_duration']:.3f}s")
        except Exception as e:
            print(f"Error processing NeMo TextGrid: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"NeMo TextGrid not found: {nemo_path}")
    
    # Create summary table
    if all_results:
        summary_data = []
        for r in all_results: # Will only contain NeMo results
            ea = r['entire_audio']
            w15 = r['window_15s']
            sent = r['sentence']
            word = r['word']
            
            summary_data.append({
                'model': r['model'],
                'time_s': r['processing_time'], # Will be 0 for NeMo
                # Entire audio
                'entire_ARR': ea['ARR'],
                'entire_MAE_start': ea['MAE_start'],
                'entire_MAE_end': ea['MAE_end'],
                'entire_MAE_duration': ea['MAE_duration'],
                'entire_RMSE_start': ea['RMSE_start'],
                'entire_RMSE_end': ea['RMSE_end'],
                'entire_RMSE_duration': ea['RMSE_duration'],
                # 15s windows
                'win15s_mean_ARR': w15.get('mean_ARR', 0),
                'win15s_mean_MAE_start': w15.get('mean_MAE_start', float('inf')),
                'win15s_mean_MAE_duration': w15.get('mean_MAE_duration', float('inf')),
                # Sentences
                'sent_mean_ARR': sent.get('mean_ARR', 0),
                'sent_mean_MAE_start': sent.get('mean_MAE_start', float('inf')),
                'sent_mean_MAE_duration': sent.get('mean_MAE_duration', float('inf')),
                # Words
                'word_mean_start_error': word.get('mean_start_error', float('inf')),
                'word_mean_duration_error': word.get('mean_duration_error', float('inf')),
                'word_median_start_error': word.get('median_start_error', float('inf')),
                'word_max_start_error': word.get('max_start_error', float('inf')),
            })
        
        # Save and display
        df = pd.DataFrame(summary_data)
        output_csv_path = output_dir / "nemo_evaluation_summary.csv" # Adjusted filename
        df.to_csv(output_csv_path, index=False)
        
        print("\n" + "="*130)
        print("NEMO EVALUATION SUMMARY")
        print("="*130)
        # Adjusted print format for clarity with fewer columns needed for single model
        print(f"{'Model':<15} {'Level':<10} {'ARR':<7} {'MAE_start':<11} {'MAE_end':<11} {'MAE_duration':<14} {'RMSE_start':<12} {'RMSE_end':<12} {'RMSE_duration':<13}")
        print("-" * 130)
        
        for _, row in df.iterrows():
            # Entire audio
            print(f"{row['model']:<15} {'Entire':<10} {row['entire_ARR']:.3f} {row['entire_MAE_start']:.3f} {row['entire_MAE_end']:.3f} {row['entire_MAE_duration']:.3f} {row['entire_RMSE_start']:.3f} {row['entire_RMSE_end']:.3f} {row['entire_RMSE_duration']:.3f}")
            
            # 15s windows
            mae_start_15s_str = f"{row['win15s_mean_MAE_start']:.3f}" if row['win15s_mean_MAE_start'] != float('inf') else "N/A"
            mae_dur_15s_str = f"{row['win15s_mean_MAE_duration']:.3f}" if row['win15s_mean_MAE_duration'] != float('inf') else "N/A"
            print(f"{'':<15} {'15s-avg':<10} {row['win15s_mean_ARR']:.3f} {mae_start_15s_str:<11} {'--':<11} {mae_dur_15s_str:<14} {'--':<12} {'--':<12} {'--':<13}")

            # Sentences
            mae_start_sent_str = f"{row['sent_mean_MAE_start']:.3f}" if row['sent_mean_MAE_start'] != float('inf') else "N/A"
            mae_dur_sent_str = f"{row['sent_mean_MAE_duration']:.3f}" if row['sent_mean_MAE_duration'] != float('inf') else "N/A"
            print(f"{'':<15} {'Sent-avg':<10} {row['sent_mean_ARR']:.3f} {mae_start_sent_str:<11} {'--':<11} {mae_dur_sent_str:<14} {'--':<12} {'--':<12} {'--':<13}")

            # Words (Mean errors)
            word_start_err_str = f"{row['word_mean_start_error']:.3f}" if row['word_mean_start_error'] != float('inf') else "N/A"
            word_dur_err_str = f"{row['word_mean_duration_error']:.3f}" if row['word_mean_duration_error'] != float('inf') else "N/A"
            # Word (Median and Max errors) - can add more columns or a separate print if needed
            word_median_start_err_str = f"{row['word_median_start_error']:.3f}" if row['word_median_start_error'] != float('inf') else "N/A"
            word_max_start_err_str = f"{row['word_max_start_error']:.3f}" if row['word_max_start_error'] != float('inf') else "N/A"
            
            print(f"{'':<15} {'Word-avg':<10} {'--':<7} {word_start_err_str:<11} {'--':<11} {word_dur_err_str:<14} {'--':<12} {'--':<12} {'--':<13}")
            print(f"{'':<15} {'Word-med':<10} {'--':<7} {word_median_start_err_str:<11} {'--':<11} {'--':<14} {'--':<12} {'--':<12} {'--':<13}")
            print(f"{'':<15} {'Word-max':<10} {'--':<7} {word_max_start_err_str:<11} {'--':<11} {'--':<14} {'--':<12} {'--':<12} {'--':<13}")

            print() # Newline for readability between models (though only one here)
            
        print(f"Results saved to: {output_csv_path}")
    else:
        print("No results were generated for NeMo.")

if __name__ == "__main__":
    run_nemo_evaluation()