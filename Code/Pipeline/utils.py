from pathlib import Path
import re
import glob

def extract_clean_text_from_textgrid(textgrid_content: str) -> str:
    """
    Extracts and cleans the transcribed text from a TextGrid file content.

    This function looks for lines that contain 'text =', extracts the text value,
    removes annotations (e.g., [laugh]) and punctuation like commas and semicolons,
    and returns a cleaned string of all the text segments joined by spaces.

    Args:
        textgrid_content (str): The content of a TextGrid file as a string.

    Returns:
        str: A single string containing the cleaned and concatenated text segments.
    """
    lines = textgrid_content.split('\n')
    text_segments = []
    for line in lines:
        if 'text = ' in line:
            text = line.split('=')[1].strip().strip('"')
            if text and text != " ":
                text = re.sub(r'\[.*?\]', '', text)
                text = text.replace(',', '').replace(';', '')
                text_segments.append(text)
    return ' '.join(text_segments)

def save_clean_transcriptions_from_textgrids(input_dir: Path, output_dir: Path) -> None:
    """
    Processes all TextGrid files in a given directory and writes extracted 
    transcriptions to text files in an output directory.

    For each .TextGrid file in the input directory, the function extracts the
    transcription using `parse_textgrid` and saves the result as a .txt file
    with the same basename in the output directory.

    Args:
        input_dir (Path): Path to the directory containing .TextGrid files.
        output_dir (Path): Path to the directory where output .txt files will be saved.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for textgrid_path in glob.glob(str(input_dir / '*.[Tt][Ee][Xx][Tt][Gg][Rr][Ii][Dd]')):
        try:
            content = Path(textgrid_path).read_text(encoding='utf-8')
            transcription = extract_clean_text_from_textgrid(content)
            output_path = output_dir / (Path(textgrid_path).stem + ".txt")
            output_path.write_text(transcription, encoding='utf-8')
            # print(f"Processed: {textgrid_path} -> {output_path}")
        except Exception as e:
            print(f"Error processing {textgrid_path}: {e}")