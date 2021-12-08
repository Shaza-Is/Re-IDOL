import os

from dotenv import load_dotenv
from app.api.command import CommandLine

def main():
    CommandLine()


if __name__ == "__main__":
    load_dotenv()
    main()