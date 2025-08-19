#!/bin/bash

# A simple script to remove a specific first line from Python files.

# Check if a directory was provided as a command-line argument.
if [ -z "$1" ]; then
    echo "Error: Please provide a directory path as an argument."
    echo "Usage: $0 <directory_path>"
    exit 1
fi

# Store the provided directory path.
target_dir="$1"

# Check if the provided directory exists.
if [ ! -d "$target_dir" ]; then
    echo "Error: Directory not found - '$target_dir'."
    exit 1
fi

echo "Processing Python files in: $target_dir"

# Loop through all files in the specified directory ending with .py
for file in "$target_dir"/*.py; do
    # Check if the file pattern matched any files.
    # This prevents the script from running the sed command on the literal string "*.py"
    # if no Python files are found.
    if [ -e "$file" ]; then
        # Get the file's name without the extension (e.g., 'losses' from 'losses.py')
        filename=$(basename "$file" .py)

        # Use sed to check and remove the specific line.
        # The '1' addresses the first line, and the '/ ... /d' is a pattern-based deletion.
        # We escape the dot '.' because it's a special character in regular expressions.
        # The -i flag edits the file in place.
        sed -i "1s/^# ${filename}\.py$//g" "$file"
        
        echo "Cleaned: $file"
    fi
done

echo "Script complete. First lines have been checked and removed from Python files."