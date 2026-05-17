"""Basic computation tools: calculator and Python code execution."""
import os
import subprocess
import tempfile

from langchain_core.tools import tool


@tool
def calculator(a: float, b: float, type: str) -> float:
    """Performs mathematical calculations, addition, subtraction, multiplication, division, modulus.
    Args:
        a (float): first float number
        b (float): second float number
        type (str): the type of calculation to perform, can be addition, subtraction, multiplication, division, modulus
    """
    if type == "addition":
        return a + b
    elif type == "subtraction":
        return a - b
    elif type == "multiplication":
        return a * b
    elif type == "division":
        if b == 0:
            raise ValueError("Cannot divide by zero.")
        return a / b
    elif type == "modulus":
        return a % b
    else:
        raise TypeError(f"{type} is not an option for type, choose one of addition, subtraction, multiplication, division, modulus")


@tool
def python_eval(code: str) -> str:
    """
    Execute a Python code snippet and return its stdout output.
    Use this when a question asks what a script outputs, or when computation requires running code.

    Args:
        code: Python source code to execute.

    Returns:
        The stdout output of the code, or an error/timeout message.
    """
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            tmp_path = f.name
        result = subprocess.run(
            ['python3', tmp_path],
            capture_output=True, text=True, timeout=30
        )
        os.unlink(tmp_path)
        if result.returncode == 0:
            return f"Output:\n{result.stdout}"
        return f"[python_eval] exit {result.returncode}:\n{result.stderr}"
    except subprocess.TimeoutExpired:
        return "[python_eval] execution timed out (30s limit)"
    except Exception as e:
        return f"[python_eval] failed: {e}"
