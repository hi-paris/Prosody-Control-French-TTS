import os
import subprocess
import sys
from pathlib import Path
import shutil
import logging # Import the logging module

# Get a logger for this module
logger = logging.getLogger(__name__)

base_dir = Path(os.path.dirname(os.path.abspath(__file__))).parent.absolute().parent.absolute().__str__()

def is_cuda_available():
    """Check if CUDA is available using torch"""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        # If torch is not available, assume CUDA is not available
        logger.warning("torch not found, assuming CUDA is not available.")
        return False

def main(input_file, output_file):
    logger.info(f"Applying demucs on {input_file}...")

    # --- Find the demucs executable ---
    demucs_executable = shutil.which("demucs")
    if not demucs_executable:
        logger.error("'demucs' executable not found in PATH.")
        logger.error("Please ensure demucs is installed and its location is in your PATH environment variable,")
        logger.error("or hardcode the full path to the executable in demucs_process.py.")
        # Fallback to copying original file if demucs not found
        logger.warning("Using original file as fallback.")
        try:
            shutil.copy(input_file, output_file)
            logger.info(f"Copied original file to {output_file}")
        except Exception as copy_e:
             logger.error(f"Error copying original file: {str(copy_e)}")
        return # Exit function
    logger.info(f"Using demucs executable found at: {demucs_executable}")

    # Detect if CUDA is available and use appropriate device
    use_cpu = not is_cuda_available()
    if use_cpu:
        logger.info("No CUDA detected or torch not available. Using CPU for processing...")
        # Use the full path here
        demucs_command = [demucs_executable, "--device", "cpu", input_file]
    else:
        logger.info("CUDA detected. Using GPU for processing...")
        # Use the full path here
        demucs_command = [demucs_executable, input_file]

    # Run Demucs
    try:
        logger.info(f"Running demucs command: {' '.join(demucs_command)}")
        result = subprocess.run(demucs_command, capture_output=True, text=True, check=False)
        
        # Check for CUDA out of memory error
        if result.returncode != 0:
            logger.error(f"Demucs command failed with return code {result.returncode}")
            logger.error(f"Stderr: {result.stderr}")
            logger.error(f"Stdout: {result.stdout}")
            
            if "CUDA error: out of memory" in result.stderr:
                logger.warning("CUDA out of memory error detected. Retrying with CPU...")
                # Retry with CPU, using the full path
                cpu_command = [demucs_executable, "--device", "cpu", input_file]
                logger.info(f"Running demucs command (CPU retry): {' '.join(cpu_command)}")
                result = subprocess.run(cpu_command, capture_output=True, text=True, check=False)
            
            # If still failed, use original file
            if result.returncode != 0:
                logger.error("Demucs processing failed even after CPU retry (or no retry attempted). Using original file as fallback.")
                shutil.copy(input_file, output_file)
                logger.info(f"Copied original file to {output_file}")
                return  # Exit function instead of sys.exit(0)
        else:
             logger.info("Demucs command executed successfully.")
             logger.debug(f"Demucs stdout: {result.stdout}") # Log stdout only on success at debug level
    
    except FileNotFoundError:
         # This specific error might occur if the path is wrong even with shutil.which or hardcoding
         logger.error(f"Error: The specified demucs executable was not found at '{demucs_executable}'.")
         logger.warning("Using original file as fallback.")
         shutil.copy(input_file, output_file)
         logger.info(f"Copied original file to {output_file}")
         return # Exit function
    except Exception as e:
        logger.error(f"Error running Demucs: {str(e)}", exc_info=True) # Log traceback
        logger.warning("Using original file as fallback.")
        shutil.copy(input_file, output_file)
        logger.info(f"Copied original file to {output_file}")
        return  # Exit function instead of sys.exit(0)

    # Get audio base name
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    logger.info(f"Base name: {base_name}")

    # --- Modified Demucs Command to specify output directory ---
    output_dir_demucs = Path(base_dir) / "separated" # Define a predictable output base
    output_dir_demucs.mkdir(parents=True, exist_ok=True) # Ensure it exists

    if use_cpu:
        demucs_command = [demucs_executable, "--device", "cpu", "-o", str(output_dir_demucs), input_file]
    else:
        demucs_command = [demucs_executable, "-o", str(output_dir_demucs), input_file]

    # --- Re-run Demucs with specified output ---
    try:
        logger.info(f"Running demucs command with output dir: {' '.join(demucs_command)}")
        result = subprocess.run(demucs_command, capture_output=True, text=True, check=False)

        if result.returncode != 0:
            logger.error(f"Demucs command failed with return code {result.returncode}")
            logger.error(f"Stderr: {result.stderr}")
            logger.error(f"Stdout: {result.stdout}")

            if "CUDA error: out of memory" in result.stderr:
                logger.warning("CUDA out of memory error detected. Retrying with CPU...")
                cpu_command = [demucs_executable, "--device", "cpu", "-o", str(output_dir_demucs), input_file]
                logger.info(f"Running demucs command (CPU retry): {' '.join(cpu_command)}")
                result = subprocess.run(cpu_command, capture_output=True, text=True, check=False)

            if result.returncode != 0:
                logger.error("Demucs processing failed. Using original file as fallback.")
                shutil.copy(input_file, output_file)
                logger.info(f"Copied original file to {output_file}")
                return
        else:
            logger.info("Demucs command executed successfully.")
            logger.debug(f"Demucs stdout: {result.stdout}")

    except FileNotFoundError:
         logger.error(f"Error: The specified demucs executable was not found at '{demucs_executable}'.")
         logger.warning("Using original file as fallback.")
         shutil.copy(input_file, output_file)
         logger.info(f"Copied original file to {output_file}")
         return
    except Exception as e:
        logger.error(f"Error running Demucs: {str(e)}", exc_info=True)
        logger.warning("Using original file as fallback.")
        shutil.copy(input_file, output_file)
        logger.info(f"Copied original file to {output_file}")
        return

    # --- Locate the output file ---
    demucs_output_path = output_dir_demucs / "htdemucs" / base_name / "vocals.wav"
    logger.info(f"Checking for Demucs output at: {demucs_output_path}")

    if not demucs_output_path.exists():
        logger.error(f"Could not find Demucs output file at expected location: {demucs_output_path}")
        # Check if the 'separated' directory contains *any* model output
        possible_models = list(output_dir_demucs.glob("*"))
        if possible_models:
             logger.warning(f"Found model directories in {output_dir_demucs}: {possible_models}. Expected 'htdemucs'.")
             # Try to find vocals.wav in the first found model dir as a guess
             first_model_dir = next((d for d in possible_models if d.is_dir()), None)
             if first_model_dir:
                 potential_output = first_model_dir / base_name / "vocals.wav"
                 if potential_output.exists():
                     logger.warning(f"Found vocals.wav in unexpected model dir: {potential_output}. Using this.")
                     demucs_output_path = potential_output
                 else:
                     logger.error(f"Could not find vocals.wav in {potential_output} either.")
                     demucs_output_path = None # Reset path if not found
             else:
                 demucs_output_path = None
        else:
             logger.error(f"No model output directories found within {output_dir_demucs}.")
             demucs_output_path = None # Reset path

        # Fallback if still not found
        if not demucs_output_path or not demucs_output_path.exists():
            logger.warning("Using original file as fallback.")
            shutil.copy(input_file, output_file)
            logger.info(f"Copied original file to {output_file}")
            return
    else:
        logger.info(f"Found Demucs output at: {demucs_output_path}")


    # Copy the demucs output to the desired output location
    try:
        shutil.copy(str(demucs_output_path), output_file) # Use str() for compatibility if needed
        logger.info(f"Demucs processing completed. Output saved to {output_file}")
        # Optional: Clean up the demucs output directory
        # shutil.rmtree(output_dir_demucs / "htdemucs" / base_name)
        # logger.info(f"Cleaned up demucs output directory: {output_dir_demucs / 'htdemucs' / base_name}")
    except Exception as e:
        logger.error(f"Error copying Demucs output: {str(e)}", exc_info=True)
        logger.warning("Using original file as fallback.")
        shutil.copy(input_file, output_file)
        logger.info(f"Copied original file to {output_file}")

if __name__ == "__main__":
    # Basic logging setup for standalone execution
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    if len(sys.argv) != 3:
        logger.error("Usage: python demucs_process.py <input_file> <output_file>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    # Ensure input file exists before proceeding
    if not os.path.exists(input_file):
        logger.error(f"Input file not found: {input_file}")
        sys.exit(1)
        
    main(input_file, output_file)