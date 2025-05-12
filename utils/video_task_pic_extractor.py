import os
import json
import cv2
import numpy as np
from pathlib import Path

# Import the constants from analysis_utils.py
from data_processing.analysis_utils import (
    FPS,
    TASK_1_PIC_START, TASK_2_PIC_START, TASK_3_PIC_START, TASK_4_PIC_START,
    TASK_5_PIC_START, TASK_6_PIC_START, TASK_7_PIC_START, TASK_8_PIC_START, TASK_9_PIC_START
)

class VideoFrameExtractor:
    def __init__(self, json_path, output_dir="extracted_frames"):
        """
        Initialize the VideoFrameExtractor.
        
        Args:
            json_path (str): Path to the JSON file containing video metadata
            output_dir (str): Directory to save extracted frames
        """
        self.json_path = json_path
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Load the JSON data
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        
        # Create a mapping from task number to timestamps
        self.task_pic_timestamps = {
            "task0": TASK_1_PIC_START,
            "task1": TASK_2_PIC_START,
            "task2": TASK_3_PIC_START,
            "task3": TASK_4_PIC_START,
            "task4": TASK_5_PIC_START,
            "task5": TASK_6_PIC_START,
            "task6": TASK_7_PIC_START,
            "task7": TASK_8_PIC_START,
            "task8": TASK_9_PIC_START,
        }
        
    def get_task_pic_pairs(self):
        """
        Extract all unique (task_no, pic_no) pairs from the JSON file.
        
        Returns:
            list: List of (task_no, pic_no) tuples
        """
        task_pic_pairs = []
        for subject_id, subject_data in self.data.items():
            if subject_data['data_type'] == 'resting':
                continue
            if 'task_no' in subject_data and 'pic_no' in subject_data:
                task_pic = (subject_data['task_no'], subject_data['pic_no'])
                if task_pic not in task_pic_pairs:
                    task_pic_pairs.append(task_pic)
        
        return task_pic_pairs
    
    def extract_frame(self, video_path, timestamp, output_filename):
        """
        Extract a single frame from a video at the specified timestamp.
        
        Args:
            video_path (str): Path to the video file
            timestamp (int): Timestamp in seconds
            output_filename (str): Name of the output file
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return False
        
        # Convert timestamp to frame number
        ### TODO: task5 needs multiple frames to track the movement
        frame_number = int(timestamp * FPS) + 2 
        
        # Set the position
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        
        # Read the frame
        ret, frame = cap.read()
        if not ret:
            print(f"Error: Could not read frame at timestamp {timestamp}")
            cap.release()
            return False
        
        # Save the frame
        output_path = os.path.join(self.output_dir, output_filename)
        cv2.imwrite(output_path, frame)
        print(f"Saved frame to {output_path}")
        
        cap.release()
        return True
        
    def extract_all_frames(self, video_path):
        """
        Extract frames for all task-pic pairs from the single video file.
        
        Args:
            video_path (str): Path to the video file
        """
        # Get all task-pic pairs
        task_pic_pairs = self.get_task_pic_pairs()
        print(f"Found {len(task_pic_pairs)} unique task-pic pairs")
        
        # Check if video exists
        if not os.path.exists(video_path):
            print(f"Error: Video file not found at {video_path}")
            return
        
        # Extract frames for each task-pic pair
        for task_no, pic_no in task_pic_pairs:
            # Get the task number (without "task" prefix)
            task_num = int(task_no[4:])
            
            # Get the picture number (without "pic" prefix)
            pic_num = int(pic_no[3:])
            
            # Get the timestamp for this task-pic pair
            if task_num < 0 or task_num > 9:
                print(f"Invalid task number: {task_no}")
                continue
            
            task_timestamps = self.task_pic_timestamps[f"task{task_num}"]
            
            if pic_num < 0 or pic_num >= len(task_timestamps):
                print(f"Warning: Picture number {pic_no} for task {task_no} is out of range. Skipping.")
                continue
            
            timestamp = task_timestamps[pic_num]
    
            output_filename = f"video_{task_no}_{pic_no}.jpg"
            self.extract_frame(video_path, timestamp, output_filename)


def main():
    # Path to the JSON file
    json_path = "/home/mmWave_group/EasyCog/data_json_files/ASR_ASR_EOG_resampleET_new.json"
    
    output_dir = "/home/mmWave_group/EasyCog/data_collection/video_task_frames"

    # Path to the single video file used for all subjects
    video_path = "/home/mmWave_group/EasyCog/data_collection/video_task_v20241103.mp4"
    
    # Create the extractor
    extractor = VideoFrameExtractor(json_path, output_dir)
    
    # Extract all frames
    extractor.extract_all_frames(video_path)
    
    print("Frame extraction complete")

if __name__ == "__main__":
    main()