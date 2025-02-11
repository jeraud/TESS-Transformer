#!/usr/bin/env python3
import os
import shutil

def flatten_directory(src_dir, dest_dir):
    # Create the destination directory if it doesn't exist.
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    
    # Walk through the source directory tree.
    for root, dirs, files in os.walk(src_dir):
        for filename in files:
            source_path = os.path.join(root, filename)
            destination_path = os.path.join(dest_dir, filename)
            # # Handle duplicate filenames by renaming the file.
            # if os.path.exists(destination_path):
            #     base, ext = os.path.splitext(filename)
            #     counter = 1
            #     new_filename = f"{base}_{counter}{ext}"
            #     destination_path = os.path.join(dest_dir, new_filename)
            #     while os.path.exists(destination_path):
            #         counter += 1
            #         new_filename = f"{base}_{counter}{ext}"
            #         destination_path = os.path.join(dest_dir, new_filename)
            
            # Copy the file, including its metadata.
            shutil.copy2(source_path, destination_path)
            # print(f"Copied '{source_path}' to '{destination_path}'")

if __name__ == "__main__":
    source_directory = "/Users/paul/Desktop/UROP/data/s0015/0000/"  # Change to your nested directory path.
    destination_directory = "/Users/paul/Desktop/UROP/data/sector_15_curves/"  # Change to your flat directory path.
    flatten_directory(source_directory, destination_directory)
    print(len(os.listdir(destination_directory)))
