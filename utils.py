import yaml
from pathlib import Path
from langchain_core.messages import SystemMessage


def load_prompt(prompt_location: str) -> SystemMessage:
    """Load system prompt from YAML file."""
    with open(prompt_location) as f:
        try:
            prompt = yaml.safe_load(f)["prompt"]
            return SystemMessage(content=prompt)
        except yaml.YAMLError as exc:
            print(exc)
            return SystemMessage(content="You are a helpful assistant.")


def get_file_extension(file_path: str) -> str:
    """Extract file extension from path."""
    return Path(file_path).suffix.lower()