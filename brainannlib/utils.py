from dotenv import load_dotenv
import os

load_dotenv()

def get_root_dir():
    return os.getenv("ALGONAUTS_ROOT_DIR")

def get_hf_dir():
    return os.getenv("HF_HOME")

def get_output_actvations_dir():
    return os.getenv("ALGONAUTS_ACTIVATIONS_DIR")

def color_print(*args, color=None):
    colors = {
        'red': '\033[91m',
        'green': '\033[92m',
        'blue': '\033[94m',
        'yellow': '\033[93m',
        'magenta': '\033[95m',
        'cyan': '\033[96m',
        'white': '\033[97m'  # Added white
    }
    # Use white if color is None or not in the dictionary
    color_code = colors.get(color, colors['white'])
    # Convert all arguments to strings and join them with spaces
    text = ' '.join(str(arg) for arg in args)
    return print(f"{color_code}{text}\033[0m")