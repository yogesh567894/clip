#!/usr/bin/env python
"""
Cleanup Tool - Clean up old output files to free up disk space
"""

import os
import sys
import argparse
import glob
import time
import shutil
from datetime import datetime

def get_dir_size(path):
    """Calculate total directory size in MB"""
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.exists(fp):
                total_size += os.path.getsize(fp)
    return total_size / (1024 * 1024)  # Convert to MB

def get_available_space(path):
    """Get available disk space in MB"""
    if os.name == 'nt':  # Windows
        try:
            import ctypes
            free_bytes = ctypes.c_ulonglong(0)
            ctypes.windll.kernel32.GetDiskFreeSpaceExW(ctypes.c_wchar_p(path), 
                                                     None, None, ctypes.pointer(free_bytes))
            return free_bytes.value / (1024 * 1024)  # Convert to MB
        except:
            return None
    else:  # Unix-like
        try:
            st = os.statvfs(path)
            return (st.f_bavail * st.f_frsize) / (1024 * 1024)  # Convert to MB
        except:
            return None

def cleanup_directory(directory, days_old=7, pattern="*.mp4", min_free_space=1000, force=False):
    """Clean up files older than specified days"""
    if not os.path.exists(directory):
        print(f"Directory '{directory}' does not exist.")
        return False
        
    # Get current free space
    free_space = get_available_space(directory)
    dir_size = get_dir_size(directory)
    
    print(f"Directory: {directory}")
    print(f"Size: {dir_size:.1f} MB")
    
    if free_space is not None:
        print(f"Free space: {free_space:.1f} MB")
        if free_space > min_free_space and not force:
            print(f"You have sufficient disk space (> {min_free_space} MB). Use --force to clean anyway.")
            return True
    
    # Get files matching pattern
    files = glob.glob(os.path.join(directory, pattern))
    
    if not files:
        print(f"No files matching '{pattern}' found in '{directory}'.")
        return False
        
    # Sort files by modification time (oldest first)
    files.sort(key=os.path.getmtime)
    
    now = time.time()
    deleted_count = 0
    deleted_size = 0
    
    print(f"\nFound {len(files)} files matching '{pattern}'")
    
    # Create backup list
    backup_list = []
    
    for file in files:
        mod_time = os.path.getmtime(file)
        age_days = (now - mod_time) / (24 * 3600)
        size_mb = os.path.getsize(file) / (1024 * 1024)
        
        if age_days > days_old or force:
            backup_list.append((file, age_days, size_mb))
    
    # Show files to be deleted
    if backup_list:
        print(f"\nFiles to clean up (older than {days_old} days):")
        for i, (file, age, size) in enumerate(backup_list):
            print(f"{i+1}. {os.path.basename(file)}")
            print(f"   Age: {age:.1f} days, Size: {size:.1f} MB")
            
        if not force:
            confirm = input("\nProceed with cleanup? (y/n): ").lower()
            if confirm != 'y':
                print("Cleanup cancelled.")
                return False
        
        # Create backup directory
        backup_dir = os.path.join(directory, "old_backups")
        if not os.path.exists(backup_dir):
            try:
                os.makedirs(backup_dir)
                print(f"Created backup directory: {backup_dir}")
            except Exception as e:
                print(f"Could not create backup directory: {e}")
                backup_dir = None
        
        # Delete or move files
        for file, age, size in backup_list:
            try:
                if backup_dir and os.path.exists(backup_dir):
                    # Try to move to backup first
                    backup_file = os.path.join(backup_dir, os.path.basename(file))
                    try:
                        shutil.move(file, backup_file)
                        print(f"Moved: {os.path.basename(file)} to backup")
                    except Exception as e:
                        # If move fails, delete
                        os.remove(file)
                        print(f"Deleted: {os.path.basename(file)}")
                else:
                    # No backup, just delete
                    os.remove(file)
                    print(f"Deleted: {os.path.basename(file)}")
                
                deleted_count += 1
                deleted_size += size
            except Exception as e:
                print(f"Error processing {file}: {e}")
        
        print(f"\nCleanup complete. Removed {deleted_count} files ({deleted_size:.1f} MB)")
        
        # Check new free space
        new_free_space = get_available_space(directory)
        if new_free_space is not None:
            print(f"Free space now: {new_free_space:.1f} MB")
            
        return True
    else:
        print(f"No files older than {days_old} days found.")
        return True

def main():
    parser = argparse.ArgumentParser(
        description="Cleanup Tool - Remove old output files to free up disk space"
    )
    
    parser.add_argument("directory", nargs="?", default="output_videos",
                        help="Directory to clean (default: output_videos)")
    parser.add_argument("--days", type=int, default=7,
                        help="Delete files older than this many days (default: 7)")
    parser.add_argument("--pattern", default="*.mp4",
                        help="File pattern to match (default: *.mp4)")
    parser.add_argument("--min-space", type=int, default=1000,
                        help="Minimum free space in MB before cleaning (default: 1000)")
    parser.add_argument("--force", action="store_true",
                        help="Force cleanup without confirmation")
    parser.add_argument("--all", action="store_true",
                        help="Clean all output directories")
    
    args = parser.parse_args()
    
    # Clean all output directories if requested
    if args.all:
        directories = ["output_videos", "output_clips", "fixed_videos"]
        success = True
        for directory in directories:
            if os.path.exists(directory):
                print(f"\nProcessing directory: {directory}")
                result = cleanup_directory(
                    directory, 
                    days_old=args.days, 
                    pattern=args.pattern,
                    min_free_space=args.min_space,
                    force=args.force
                )
                success = success and result
            else:
                print(f"Directory '{directory}' does not exist, skipping.")
        return 0 if success else 1
    else:
        # Clean single directory
        result = cleanup_directory(
            args.directory, 
            days_old=args.days, 
            pattern=args.pattern,
            min_free_space=args.min_space,
            force=args.force
        )
        return 0 if result else 1

if __name__ == "__main__":
    sys.exit(main()) 