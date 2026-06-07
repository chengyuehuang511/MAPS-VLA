"""
Important constants for VLA training and evaluation.

Attempts to automatically identify the correct constants to set based on the Python command used to launch
training or evaluation. If it is unclear, defaults to using the LIBERO simulation benchmark constants.
"""
import sys
from enum import Enum

IGNORE_INDEX = -100
STOP_INDEX = 2  # '</s>'
# NUM_TOKENS = 64


# Defines supported normalization schemes for action and proprioceptive state.
class NormalizationType(str, Enum):
    # fmt: off
    NORMAL = "normal"               # Normalize to Mean = 0, Stdev = 1
    BOUNDS = "bounds"               # Normalize to Interval = [-1, 1]
    BOUNDS_Q99 = "bounds_q99"       # Normalize [quantile_01, ..., quantile_99] --> [-1, ..., 1]
    # fmt: on


# Define constants for each robot platform
LIBERO_CONSTANTS = {
    "NUM_ACTIONS_CHUNK": 8,
    "ACTION_DIM": 7,
    "PROPRIO_DIM": 8,
    "ACTION_PROPRIO_NORMALIZATION_TYPE": NormalizationType.BOUNDS_Q99,
}

CALVIN_CONSTANTS = {
    "NUM_ACTIONS_CHUNK": 8,
    "ACTION_DIM": 7,
    "PROPRIO_DIM": 8,
    "ACTION_PROPRIO_NORMALIZATION_TYPE": NormalizationType.BOUNDS_Q99,
}

ALOHA_CONSTANTS = {
    "NUM_ACTIONS_CHUNK": 25,
    "ACTION_DIM": 14,
    "PROPRIO_DIM": 14,
    "ACTION_PROPRIO_NORMALIZATION_TYPE": NormalizationType.BOUNDS,
}

BRIDGE_CONSTANTS = {
    "NUM_ACTIONS_CHUNK": 5,
    "ACTION_DIM": 7,
    "PROPRIO_DIM": 7,
    "ACTION_PROPRIO_NORMALIZATION_TYPE": NormalizationType.BOUNDS_Q99,
}


# Function to detect robot platform from command line arguments
def detect_robot_platform():
    cmd_args = " ".join(sys.argv).lower()

    if "libero" in cmd_args:
        return "LIBERO"
    elif "aloha" in cmd_args:
        return "ALOHA"
    elif "bridge" in cmd_args:
        return "BRIDGE"
    elif "calvin" in cmd_args:
        return "CALVIN"
    else:
        # Default to LIBERO if unclear
        return "LIBERO"

def detect_model():
    cmd_args = " ".join(sys.argv).lower()
    print(f"cmd_args: {cmd_args}")

    if "prism-dinosiglip-224px+7b" in cmd_args or "openvla-oft" in cmd_args:
        return "LLAMA_2"
    elif "prism-qwen25-extra-dinosiglip-224px-0_5b" in cmd_args:
        return "QWEN_2_5"
    else:
        # Default to QWEN_2_5 if unclear
        return "QWEN_2_5"


# Determine which robot platform to use
ROBOT_PLATFORM = detect_robot_platform()
MODEL_TYPE = detect_model()

# Set the appropriate constants based on the detected platform
if ROBOT_PLATFORM == "LIBERO":
    constants = LIBERO_CONSTANTS
elif ROBOT_PLATFORM == "ALOHA":
    constants = ALOHA_CONSTANTS
elif ROBOT_PLATFORM == "BRIDGE":
    constants = BRIDGE_CONSTANTS
elif ROBOT_PLATFORM == "CALVIN":
    constants = CALVIN_CONSTANTS

if MODEL_TYPE == "LLAMA_2":
    # Llama 2 token constants
    ACTION_TOKEN_BEGIN_IDX = 31743
elif MODEL_TYPE == "QWEN_2_5":
    # Qwen2.5-0.5B token constants
    ACTION_TOKEN_BEGIN_IDX  = 151386 # 151536?

# Assign constants to global variables
NUM_ACTIONS_CHUNK = constants["NUM_ACTIONS_CHUNK"]
ACTION_DIM = constants["ACTION_DIM"]
PROPRIO_DIM = constants["PROPRIO_DIM"]
ACTION_PROPRIO_NORMALIZATION_TYPE = constants["ACTION_PROPRIO_NORMALIZATION_TYPE"]
NUM_TOKENS = NUM_ACTIONS_CHUNK * ACTION_DIM

# Print which robot platform constants are being used (for debugging)
print(f"Using {ROBOT_PLATFORM} constants:")
print(f"  NUM_ACTIONS_CHUNK = {NUM_ACTIONS_CHUNK}")
print(f"  ACTION_DIM = {ACTION_DIM}")
print(f"  PROPRIO_DIM = {PROPRIO_DIM}")
print(f"  ACTION_PROPRIO_NORMALIZATION_TYPE = {ACTION_PROPRIO_NORMALIZATION_TYPE}")
print(f"  ACTION_TOKEN_BEGIN_IDX = {ACTION_TOKEN_BEGIN_IDX}")
print("If needed, manually set the correct constants in `prismatic/vla/constants.py`!")
