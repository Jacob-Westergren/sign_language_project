#!/usr/bin/env python 

"""
In order to run the python script, we need to tell the shell three things:

1. That the file is a script
2. Which interpreter we want to execute the script
3. The path of said interpreter
So #!PATH interpreter

The shebang #! accomplishes (1.). The shebang begins with a # because the # character is a comment marker in many scripting languages so the 
shebang line are therefore automatically ignored by the interpreter.

The /usr/bin/env is the path to the env command which asks the system to automatically find the path to the specified interpreter. We could also
hardcode it like /usr/bin/python but different machines uses different OS and setups and such so better to automatically find it. 

In my case, the interpreter is python, but in other instances it can be java, node, etc.
When we run a specific code file in the terminal, we don't need to use usr/bin/env as we already specify the interpreter in the command, so like
python main.py
"""
import os
import cv2
from dotenv import load_dotenv

def test_func():
    # Load environment variables from .env
    load_dotenv()

    # Environment variables
    api_key = os.getenv("API_KEY")  
    os.environ["CUDA_VISIBLE_DEVICES"] = "test"

    # Display environment variables
    print("Hello Venv World")
    print("API_KEY from .env:", api_key)
    print("CUDA_VISIBLE_DEVICES:", os.getenv("CUDA_VISIBLE_DEVICES"))
