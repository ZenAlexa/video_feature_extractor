import os
import sys
import torch as th
import math
import numpy as np
import pandas as pd
import argparse
import glob
import multiprocessing
import time
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

# Import project modules
from video_loader import VideoLoader
from torch.utils.data import DataLoader
from model import get_model
from preprocessing import Preprocessing
from random_sequence_shuffler import RandomSequenceSampler
import torch.nn.functional as F

class FeatureExtractor:
    def __init__(self, args):
        self.args = args
        self.setup_paths()
        self.setup_model()
        
    def setup_paths(self):
        """Setup input and output paths"""
        # Create output directory if needed
        if hasattr(self.args, 'output_dir') and self.args.output_dir:
            os.makedirs(self.args.output_dir, exist_ok=True)
            
        # Check model path
        if not os.path.exists(self.args.resnext101_model_path):
            print(f"Error: Model file {self.args.resnext101_model_path} not found!")
            print("Please download the model file first:")
            print(f"URL: https://www.rocq.inria.fr/cluster-willow/amiech/howto100m/models/resnext101.pth")
            print(f"Save it to: {self.args.resnext101_model_path}")
            sys.exit(1)
    
    def setup_model(self):
        """Initialize the model"""
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Loading 3D-ResneXt-101 model...")
        self.preprocess = Preprocessing(self.args.type)
        self.model = get_model(self.args)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Model loaded successfully")
    
    def create_csv_from_directory(self):
        """Create a CSV file listing videos to process"""
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
        video_files = []
        
        for ext in video_extensions:
            video_files.extend(glob.glob(os.path.join(self.args.input_dir, f'*{ext}')))
        
        if not video_files:
            print(f"No video files found in {self.args.input_dir}")
            return None
        
        # Create CSV content
        csv_file = 'video_list.csv'
        data = []
        for video_path in video_files:
            video_name = os.path.basename(video_path).split('.')[0]
            output_path = os.path.join(self.args.output_dir, f"{video_name}.npy")
            data.append({
                'video_path': os.path.abspath(video_path),
                'feature_path': os.path.abspath(output_path)
            })
        
        # Save CSV
        df = pd.DataFrame(data)
        df.to_csv(csv_file, index=False)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Found {len(data)} videos to process")
        return csv_file
    
    def extract_features(self, csv_file=None):
        """Extract features from videos listed in CSV file"""
        if csv_file is None and hasattr(self.args, 'csv'):
            csv_file = self.args.csv
        elif csv_file is None:
            print("Error: No CSV file provided")
            return
        
        dataset = VideoLoader(
            csv_file,
            framerate=1 if self.args.type == '2d' else 24,
            size=224 if self.args.type == '2d' else 112,
            centercrop=(self.args.type == '3d'),
        )
        n_dataset = len(dataset)
        sampler = RandomSequenceSampler(n_dataset, 10)
        
        # Configure DataLoader for Windows compatibility
        loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,  # Set to 0 to avoid multiprocessing issues on Windows
            sampler=sampler if n_dataset > 10 else None,
        )
        
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Starting feature extraction for {n_dataset} videos")
        print("-" * 80)
        
        with th.no_grad():
            for k, data in enumerate(loader):
                start_time = time.time()
                input_file = data['input'][0]
                output_file = data['output'][0]
                
                if len(data['video'].shape) > 3:
                    # Extract filename from path for display
                    filename = os.path.basename(input_file)
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] Processing video {k+1}/{n_dataset}: {filename}")
                    
                    video = data['video'].squeeze()
                    if len(video.shape) == 4:
                        video = self.preprocess(video)
                        n_chunk = len(video)
                        
                        print(f"  - Video length: {n_chunk} chunks")
                        print(f"  - Output destination: {os.path.basename(output_file)}")
                        
                        # Use GPU if available
                        if th.cuda.is_available():
                            features = th.cuda.FloatTensor(n_chunk, 2048).fill_(0)
                        else:
                            features = th.zeros(n_chunk, 2048)
                        
                        n_iter = int(math.ceil(n_chunk / float(self.args.batch_size)))
                        
                        # Process in batches with progress bar
                        for i in tqdm(range(n_iter), desc="Extracting features", unit="batch"):
                            min_ind = i * self.args.batch_size
                            max_ind = min(n_chunk, (i + 1) * self.args.batch_size)
                            
                            # Send to GPU if available
                            if th.cuda.is_available():
                                video_batch = video[min_ind:max_ind].cuda()
                            else:
                                video_batch = video[min_ind:max_ind]
                            
                            # Extract features
                            batch_features = self.model(video_batch)
                            if self.args.l2_normalize:
                                batch_features = F.normalize(batch_features, dim=1)
                            
                            features[min_ind:max_ind] = batch_features
                        
                        # Convert to numpy and save
                        features = features.cpu().numpy()
                        if self.args.half_precision:
                            features = features.astype('float16')
                        
                        np.save(output_file, features)
                        elapsed_time = time.time() - start_time
                        fps = n_chunk / elapsed_time
                        
                        print(f"  - Features saved: shape={features.shape}")
                        print(f"  - Processing speed: {fps:.2f} chunks/second")
                        print(f"  - Time taken: {elapsed_time:.2f} seconds")
                        print("-" * 80)
                else:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] Video already processed: {os.path.basename(input_file)}")
        
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Feature extraction completed!")
        print(f"All features saved to: {self.args.output_dir if hasattr(self.args, 'output_dir') else 'specified paths'}")

def main():
    parser = argparse.ArgumentParser(description='Video Feature Extractor')
    
    # Mode selection
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--csv', type=str, help='Input CSV with video paths and output paths')
    group.add_argument('--input_dir', type=str, help='Directory containing videos to process')
    
    # Required for directory mode
    parser.add_argument('--output_dir', type=str, help='Directory for saving extracted features')
    
    # Optional parameters
    parser.add_argument('--type', type=str, default='3d', choices=['2d', '3d'],
                       help='CNN type (2d or 3d)')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for feature extraction')
    parser.add_argument('--half_precision', type=int, default=1,
                       help='Output half precision float')
    parser.add_argument('--l2_normalize', type=int, default=1,
                       help='L2 normalize features')
    parser.add_argument('--resnext101_model_path', type=str, default='model/resnext101.pth',
                       help='Path to the model file')
    
    args = parser.parse_args()
    
    # Validate args
    if args.input_dir and not args.output_dir:
        parser.error("--output_dir is required when using --input_dir")
    
    # Initialize extractor
    extractor = FeatureExtractor(args)
    
    # Process based on input mode
    if args.input_dir:
        # Directory mode - create CSV and process
        csv_file = extractor.create_csv_from_directory()
        if csv_file:
            extractor.extract_features(csv_file)
    else:
        # CSV mode - process directly
        extractor.extract_features()

if __name__ == "__main__":
    # Add Windows multiprocessing support
    multiprocessing.freeze_support()
    main() 