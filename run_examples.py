#!/usr/bin/env python3
"""
Run Oracle Datastore Agent Examples
Simple script to run all examples
"""
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from examples.examples import main

if __name__ == "__main__":
    main()


