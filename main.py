import sys

from dotenv import load_dotenv
from app.api.command import CommandLine
from typing import List

def main(): 
    CommandLine()


if __name__ == "__main__":
    load_dotenv()
    main()