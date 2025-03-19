import os
import numpy as np
import argparse
import re
from tqdm import tqdm
from scipy.signal import correlate

class SimpleMultiViewAnalyzer:
    def __init__(self, feature_dir, output_dir=None, prefix_pattern=None):
        self.feature_dir = feature_dir
        self.output_dir = output_dir or os.path.join(feature_dir, 'analysis')
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Pattern to identify related videos
        self.prefix_pattern = prefix_pattern or r'(.+?)_view\d+'
        
        # Load features
        self.features = self.load_features()
        
        # Group related videos
        self.groups = self.group_related_videos()
    
    def load_features(self):
        """Load all feature files"""
        features = {}
        feature_files = []
        
        # List all .npy files
        for file in os.listdir(self.feature_dir):
            if file.endswith('.npy'):
                feature_files.append(os.path.join(self.feature_dir, file))
        
        # Load each file
        print(f"Loading {len(feature_files)} feature files...")
        for file_path in feature_files:
            name = os.path.basename(file_path).split('.')[0]
            try:
                features[name] = np.load(file_path)
                print(f"  - Loaded {name}: shape {features[name].shape}")
            except Exception as e:
                print(f"  - Error loading {name}: {e}")
        
        return features
    
    def group_related_videos(self):
        """Group videos that belong to the same assembly session"""
        groups = {}
        pattern = re.compile(self.prefix_pattern)
        
        for name in self.features.keys():
            match = pattern.match(name)
            if match:
                group_id = match.group(1)  # Extract common prefix
                if group_id not in groups:
                    groups[group_id] = []
                groups[group_id].append(name)
        
        # Only keep groups with multiple videos
        groups = {k: v for k, v in groups.items() if len(v) > 1}
        
        print(f"Found {len(groups)} groups of related videos:")
        for group_id, members in groups.items():
            print(f"  - {group_id}: {len(members)} views - {', '.join(members)}")
        
        return groups
    
    def align_features(self, features_a, features_b, max_shift=30):
        """Align two feature sequences using cross-correlation"""
        # Compute average feature vectors
        avg_a = np.mean(features_a, axis=1)
        avg_b = np.mean(features_b, axis=1)
        
        # Compute cross-correlation
        correlation = correlate(avg_a, avg_b, mode='full')
        max_corr_idx = np.argmax(correlation)
        shift = max_corr_idx - (len(avg_a) - 1)
        
        # Limit the maximum shift
        if abs(shift) > max_shift:
            print(f"  - Warning: Large alignment shift detected ({shift}), limiting to ±{max_shift}")
            shift = max_shift if shift > 0 else -max_shift
        
        # Create aligned versions
        if shift > 0:
            aligned_a = np.pad(features_a, ((shift, 0), (0, 0)), mode='edge')[:len(features_b)]
            aligned_b = features_b
        else:
            aligned_a = features_a
            aligned_b = np.pad(features_b, ((abs(shift), 0), (0, 0)), mode='edge')[:len(features_a)]
        
        # Make sure they're the same length
        min_len = min(len(aligned_a), len(aligned_b))
        aligned_a = aligned_a[:min_len]
        aligned_b = aligned_b[:min_len]
        
        return aligned_a, aligned_b, shift
    
    def analyze_similarity(self, features_a, features_b):
        """Analyze similarity between two aligned feature sequences"""
        # Calculate cosine similarity
        similarities = []
        for i in range(len(features_a)):
            # Normalize vectors
            norm_a = np.linalg.norm(features_a[i])
            norm_b = np.linalg.norm(features_b[i])
            
            if norm_a > 0 and norm_b > 0:
                # Cosine similarity
                sim = np.dot(features_a[i], features_b[i]) / (norm_a * norm_b)
                similarities.append(sim)
            else:
                similarities.append(0)
        
        # Calculate statistics
        avg_sim = np.mean(similarities)
        min_sim = np.min(similarities)
        max_sim = np.max(similarities)
        
        return avg_sim, min_sim, max_sim
    
    def create_fused_features(self, group_id, views, output_dir):
        """Create fused features from all views"""
        print(f"Creating fused features for {group_id}...")
        
        # Get features for all views
        view_features = {view: self.features[view] for view in views}
        
        # Find the reference view (longest one)
        ref_view = max(views, key=lambda v: len(view_features[v]))
        ref_features = view_features[ref_view]
        print(f"  - Using {ref_view} as reference (length: {len(ref_features)})")
        
        # Align all views to the reference
        aligned_features = {ref_view: ref_features}
        
        for view in views:
            if view == ref_view:
                continue
                
            # Skip if lengths are too different
            if abs(len(view_features[view]) - len(ref_features)) > 100:
                print(f"  - Skipping {view} (length difference too large)")
                continue
            
            print(f"  - Aligning {view} to reference...")
            aligned_features[view], _, shift = self.align_features(
                view_features[view], ref_features)
            print(f"    - Optimal shift: {shift} frames")
        
        # Create fused features
        all_aligned = list(aligned_features.values())
        
        # Get minimum length
        min_len = min(len(f) for f in all_aligned)
        all_aligned = [f[:min_len] for f in all_aligned]
        
        # Create mean-pooled features
        mean_fused = np.mean(all_aligned, axis=0)
        
        # Create max-pooled features
        max_fused = np.max(all_aligned, axis=0)
        
        # Save fused features
        mean_path = os.path.join(output_dir, f'{group_id}_fused_mean.npy')
        max_path = os.path.join(output_dir, f'{group_id}_fused_max.npy')
        
        np.save(mean_path, mean_fused)
        np.save(max_path, max_fused)
        
        print(f"  - Saved mean-fused features: {mean_path}")
        print(f"  - Saved max-fused features: {max_path}")
        print(f"  - Feature shape: {mean_fused.shape}")
        
        return mean_fused, max_fused
    
    def analyze_group(self, group_id, views):
        """Analyze a group of related videos"""
        print(f"\nAnalyzing group: {group_id}")
        
        if len(views) < 2:
            print("Need at least 2 views for analysis")
            return
        
        # Create output directory for this group
        group_dir = os.path.join(self.output_dir, group_id)
        os.makedirs(group_dir, exist_ok=True)
        
        # Get features for all views
        view_features = {view: self.features[view] for view in views}
        
        # Analyze each pair of views
        print(f"Analyzing {len(views)} views with {len(views)*(len(views)-1)//2} comparisons")
        for i in range(len(views)):
            for j in range(i+1, len(views)):
                view_i, view_j = views[i], views[j]
                print(f"\nComparing {view_i} and {view_j}")
                
                # Skip if lengths are too different
                len_i = len(view_features[view_i])
                len_j = len(view_features[view_j])
                if abs(len_i - len_j) > 100:
                    print(f"  - Length difference too large: {len_i} vs {len_j}")
                    continue
                
                # Align features
                print(f"  - Aligning features...")
                aligned_i, aligned_j, shift = self.align_features(
                    view_features[view_i], view_features[view_j])
                print(f"  - Optimal shift: {shift} frames")
                print(f"  - Aligned length: {len(aligned_i)} frames")
                
                # Calculate similarity
                print(f"  - Calculating similarity metrics...")
                avg_sim, min_sim, max_sim = self.analyze_similarity(aligned_i, aligned_j)
                print(f"  - Similarity stats: avg={avg_sim:.4f}, min={min_sim:.4f}, max={max_sim:.4f}")
        
        # Create fused features from all views
        self.create_fused_features(group_id, views, group_dir)
        print(f"\nResults saved to: {group_dir}")
    
    def analyze_all(self):
        """Analyze all groups"""
        for group_id, views in self.groups.items():
            self.analyze_group(group_id, views)
        
        print("\nAnalysis complete!")

def main():
    parser = argparse.ArgumentParser(description='Simple Multi-view Video Feature Analyzer')
    parser.add_argument('--feature_dir', type=str, required=True,
                       help='Directory containing feature files (.npy)')
    parser.add_argument('--output_dir', type=str,
                       help='Directory for saving analysis results (default: feature_dir/analysis)')
    parser.add_argument('--pattern', type=str, default=r'(.+?)_view\d+',
                       help='Regex pattern to identify related videos (default: "(.+?)_view\\d+")')
    
    args = parser.parse_args()
    
    analyzer = SimpleMultiViewAnalyzer(
        feature_dir=args.feature_dir,
        output_dir=args.output_dir,
        prefix_pattern=args.pattern
    )
    
    analyzer.analyze_all()

if __name__ == "__main__":
    main() 