import os
import numpy as np
import argparse
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from scipy.signal import correlate
from scipy.interpolate import interp1d
import re
from tqdm import tqdm

class MultiViewAnalyzer:
    def __init__(self, feature_dir, output_dir=None, prefix_pattern=None):
        self.feature_dir = feature_dir
        self.output_dir = output_dir or os.path.join(feature_dir, 'analysis')
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Pattern to identify related videos (e.g. "assembly_1_view1", "assembly_1_view2")
        self.prefix_pattern = prefix_pattern or r'(.+?)_view\d+'
        
        # Load all features
        self.features = self.load_features()
        
        # Group related videos
        self.groups = self.group_related_videos()
        
    def load_features(self):
        """Load all feature files from the directory"""
        features = {}
        feature_files = list(Path(self.feature_dir).glob('*.npy'))
        
        for file_path in tqdm(feature_files, desc="Loading features"):
            name = file_path.stem
            try:
                features[name] = np.load(str(file_path))
            except Exception as e:
                print(f"Error loading {name}: {e}")
        
        print(f"Loaded {len(features)} feature files")
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
        
        print(f"Found {len(groups)} groups of related videos")
        for group_id, members in groups.items():
            print(f"  - {group_id}: {len(members)} views")
        
        return groups
    
    def align_features(self, features_a, features_b, max_shift=30):
        """Align two feature sequences using cross-correlation"""
        # Compute average feature vectors for simplicity
        avg_a = np.mean(features_a, axis=1)
        avg_b = np.mean(features_b, axis=1)
        
        # Compute cross-correlation to find best alignment
        correlation = correlate(avg_a, avg_b, mode='full')
        max_corr_idx = np.argmax(correlation)
        shift = max_corr_idx - (len(avg_a) - 1)
        
        # Limit the maximum shift to avoid unreasonable alignments
        if abs(shift) > max_shift:
            print(f"  - Warning: Large alignment shift detected ({shift}), limiting to ±{max_shift}")
            shift = max_shift if shift > 0 else -max_shift
        
        print(f"  - Optimal shift: {shift} frames")
        
        # Create aligned versions with padding
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
    
    def analyze_group(self, group_id, views):
        """Analyze a group of related videos"""
        print(f"\nAnalyzing group: {group_id}")
        
        if len(views) < 2:
            print("Need at least 2 views for analysis")
            return
        
        # Get features for all views
        view_features = {view: self.features[view] for view in views}
        
        # Create output directory for this group
        group_dir = os.path.join(self.output_dir, group_id)
        os.makedirs(group_dir, exist_ok=True)
        
        # Analyze each pair of views
        for i in range(len(views)):
            for j in range(i+1, len(views)):
                view_i, view_j = views[i], views[j]
                print(f"Comparing {view_i} and {view_j}")
                
                # Skip if lengths are too different
                if abs(len(view_features[view_i]) - len(view_features[view_j])) > 100:
                    print(f"  - Length difference too large: {len(view_features[view_i])} vs {len(view_features[view_j])}")
                    continue
                
                # Align features
                aligned_i, aligned_j, shift = self.align_features(
                    view_features[view_i], view_features[view_j])
                
                # Calculate similarity
                self.analyze_similarity(aligned_i, aligned_j, view_i, view_j, group_dir)
                
                # Visualize alignment
                self.visualize_alignment(aligned_i, aligned_j, view_i, view_j, group_dir)
        
        # Create combined visualization for all views in the group
        self.visualize_group(group_id, view_features, group_dir)
        
        # Create fused features from all views
        self.create_fused_features(group_id, views, group_dir)
    
    def analyze_similarity(self, features_a, features_b, name_a, name_b, output_dir):
        """Analyze similarity between two aligned feature sequences"""
        # Calculate frame-by-frame cosine similarity
        cosine_sim = np.zeros(len(features_a))
        for i in range(len(features_a)):
            cosine_sim[i] = cosine_similarity(
                features_a[i].reshape(1, -1), 
                features_b[i].reshape(1, -1)
            )[0, 0]
        
        # Plot similarity over time
        plt.figure(figsize=(12, 5))
        plt.plot(cosine_sim)
        plt.title(f'Feature Similarity: {name_a} vs {name_b}')
        plt.xlabel('Frame index')
        plt.ylabel('Cosine similarity')
        plt.grid(True)
        
        # Add horizontal lines for reference
        plt.axhline(y=0.7, color='r', linestyle='--', alpha=0.5)
        plt.axhline(y=0.8, color='g', linestyle='--', alpha=0.5)
        plt.axhline(y=0.9, color='b', linestyle='--', alpha=0.5)
        
        # Add annotations
        plt.text(len(cosine_sim)*0.02, 0.7, '0.7', color='r')
        plt.text(len(cosine_sim)*0.02, 0.8, '0.8', color='g')
        plt.text(len(cosine_sim)*0.02, 0.9, '0.9', color='b')
        
        # Save figure
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'similarity_{name_a}_vs_{name_b}.png'))
        plt.close()
        
        # Calculate statistics
        avg_sim = np.mean(cosine_sim)
        min_sim = np.min(cosine_sim)
        max_sim = np.max(cosine_sim)
        print(f"  - Similarity stats: avg={avg_sim:.4f}, min={min_sim:.4f}, max={max_sim:.4f}")
        
        return cosine_sim
    
    def visualize_alignment(self, features_a, features_b, name_a, name_b, output_dir):
        """Visualize alignment between two feature sequences"""
        # Use PCA to reduce dimensionality for visualization
        pca = PCA(n_components=10)
        
        # Stack features and fit PCA
        stacked = np.vstack([features_a, features_b])
        pca.fit(stacked)
        
        # Transform both feature sets
        pca_a = pca.transform(features_a)
        pca_b = pca.transform(features_b)
        
        # Plot first 3 components
        fig, axs = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
        
        for i in range(3):
            axs[i].plot(pca_a[:, i], label=name_a)
            axs[i].plot(pca_b[:, i], label=name_b)
            axs[i].set_ylabel(f'PC{i+1}')
            axs[i].grid(True)
            axs[i].legend()
        
        axs[2].set_xlabel('Frame index')
        plt.suptitle(f'Alignment Visualization: {name_a} vs {name_b}')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'alignment_{name_a}_vs_{name_b}.png'))
        plt.close()
    
    def visualize_group(self, group_id, view_features, output_dir):
        """Create a combined visualization for all views in a group"""
        # Use t-SNE to visualize all views together
        view_names = list(view_features.keys())
        
        # Prepare data for t-SNE
        all_features = []
        view_indices = []
        
        # Take samples at regular intervals to reduce computation
        sample_rate = max(1, min(len(next(iter(view_features.values()))) // 100, 10))
        
        for i, name in enumerate(view_names):
            features = view_features[name][::sample_rate]
            all_features.append(features)
            view_indices.extend([i] * len(features))
        
        all_features = np.vstack(all_features)
        
        # Apply t-SNE
        print(f"Computing t-SNE for group {group_id} ({len(all_features)} points)...")
        tsne = TSNE(n_components=2, random_state=42)
        embedded = tsne.fit_transform(all_features)
        
        # Plot
        plt.figure(figsize=(10, 8))
        
        # Color map for different views
        colors = plt.cm.tab10(np.linspace(0, 1, len(view_names)))
        
        for i, name in enumerate(view_names):
            mask = np.array(view_indices) == i
            plt.scatter(embedded[mask, 0], embedded[mask, 1], 
                      c=[colors[i]], label=name, alpha=0.7)
        
        plt.title(f'Feature Space Visualization for {group_id}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'tsne_{group_id}.png'))
        plt.close()
    
    def create_fused_features(self, group_id, views, output_dir):
        """Create fused features from all views"""
        print(f"Creating fused features for {group_id}...")
        
        # Get features for all views
        view_features = {view: self.features[view] for view in views}
        
        # Find the reference view (longest one)
        ref_view = max(views, key=lambda v: len(view_features[v]))
        ref_features = view_features[ref_view]
        
        # Align all views to the reference
        aligned_features = {ref_view: ref_features}
        
        for view in views:
            if view == ref_view:
                continue
                
            # Skip if lengths are too different
            if abs(len(view_features[view]) - len(ref_features)) > 100:
                print(f"  - Skipping {view} (length difference too large)")
                continue
                
            aligned_features[view], _, _ = self.align_features(
                view_features[view], ref_features)
        
        # Create fused features using mean or max
        all_aligned = [f for f in aligned_features.values()]
        
        # Get minimum length
        min_len = min(len(f) for f in all_aligned)
        all_aligned = [f[:min_len] for f in all_aligned]
        
        # Create mean-pooled features
        mean_fused = np.mean(all_aligned, axis=0)
        
        # Create max-pooled features
        max_fused = np.max(all_aligned, axis=0)
        
        # Save fused features
        np.save(os.path.join(output_dir, f'{group_id}_fused_mean.npy'), mean_fused)
        np.save(os.path.join(output_dir, f'{group_id}_fused_max.npy'), max_fused)
        
        print(f"  - Saved fused features (shape: {mean_fused.shape})")
    
    def analyze_all(self):
        """Analyze all groups"""
        for group_id, views in self.groups.items():
            self.analyze_group(group_id, views)

def main():
    parser = argparse.ArgumentParser(description='Multi-view Video Feature Analyzer')
    parser.add_argument('--feature_dir', type=str, required=True,
                       help='Directory containing feature files (.npy)')
    parser.add_argument('--output_dir', type=str,
                       help='Directory for saving analysis results (default: feature_dir/analysis)')
    parser.add_argument('--pattern', type=str, default=r'(.+?)_view\d+',
                       help='Regex pattern to identify related videos (default: "(.+?)_view\\d+")')
    
    args = parser.parse_args()
    
    analyzer = MultiViewAnalyzer(
        feature_dir=args.feature_dir,
        output_dir=args.output_dir,
        prefix_pattern=args.pattern
    )
    
    analyzer.analyze_all()

if __name__ == "__main__":
    main() 