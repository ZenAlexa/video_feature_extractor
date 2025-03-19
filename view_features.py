import os
import numpy as np
import argparse
import matplotlib.pyplot as plt
from pathlib import Path

def load_feature(file_path):
    """Load feature file and display basic information"""
    try:
        features = np.load(file_path)
        print(f"\nFeature file: {os.path.basename(file_path)}")
        print(f"Shape: {features.shape}")
        print(f"Type: {features.dtype}")
        print(f"Non-zero elements: {np.count_nonzero(features)} / {features.size} ({np.count_nonzero(features) / features.size * 100:.2f}%)")
        print(f"Mean: {np.mean(features):.6f}")
        print(f"Std Dev: {np.std(features):.6f}")
        print(f"Min: {np.min(features):.6f}")
        print(f"Max: {np.max(features):.6f}")
        return features
    except Exception as e:
        print(f"Error loading feature file: {e}")
        return None

def plot_feature_heatmap(features, output_path=None):
    """Generate feature heatmap"""
    plt.figure(figsize=(12, 8))
    
    # For visualization, only use first 100 frames and 100 dimensions
    if features.shape[0] > 100 or features.shape[1] > 100:
        sample_features = features[:100, :100]
    else:
        sample_features = features
    
    plt.imshow(sample_features, aspect='auto', cmap='viridis')
    plt.colorbar(label='Feature Value')
    plt.title(f'Feature Heatmap (First 100 frames, 100 dims)')
    plt.xlabel('Feature Dimension')
    plt.ylabel('Frame Index')
    
    if output_path:
        plt.savefig(output_path)
    plt.show()

def plot_feature_pca(features, n_components=2, output_path=None):
    """Visualize features using PCA dimensionality reduction"""
    from sklearn.decomposition import PCA
    
    # Apply PCA
    pca = PCA(n_components=n_components)
    features_reduced = pca.fit_transform(features)
    
    # Create color map
    colors = plt.cm.viridis(np.linspace(0, 1, len(features_reduced)))
    
    # Plot 2D or 3D
    if n_components == 2:
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(features_reduced[:, 0], features_reduced[:, 1], 
                   c=range(len(features_reduced)), cmap='viridis', s=5)
        plt.colorbar(scatter, label='Frame Index')
        plt.title('PCA Feature Visualization (2D)')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        
        # Add arrows to show timeline
        for i in range(0, len(features_reduced)-1, max(1, len(features_reduced)//20)):
            plt.arrow(features_reduced[i, 0], features_reduced[i, 1],
                     features_reduced[i+1, 0] - features_reduced[i, 0],
                     features_reduced[i+1, 1] - features_reduced[i, 1],
                     head_width=0.1, head_length=0.2, fc='red', ec='red', alpha=0.5)
    
    elif n_components == 3:
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(features_reduced[:, 0], features_reduced[:, 1], features_reduced[:, 2],
                  c=range(len(features_reduced)), cmap='viridis', s=5)
        plt.colorbar(scatter, label='Frame Index')
        ax.set_title('PCA Feature Visualization (3D)')
        ax.set_xlabel('Principal Component 1')
        ax.set_ylabel('Principal Component 2')
        ax.set_zlabel('Principal Component 3')
    
    if output_path:
        plt.savefig(output_path)
    plt.show()
    
    # Print explained variance ratio
    explained_var = pca.explained_variance_ratio_
    print(f"\nPCA Explained Variance Ratio:")
    for i, var in enumerate(explained_var):
        print(f"  Principal Component {i+1}: {var:.4f} ({var*100:.2f}%)")
    print(f"  Total Explained Variance: {sum(explained_var):.4f} ({sum(explained_var)*100:.2f}%)")

def plot_temporal_features(features, output_path=None):
    """Plot feature changes over time"""
    # Calculate mean features and std deviation per frame
    mean_features = np.mean(features, axis=1)
    std_features = np.std(features, axis=1)
    
    # Calculate feature change rate (first derivative)
    feature_diff = np.diff(mean_features)
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # First subplot: Mean features and std deviation
    ax1.plot(mean_features, label='Mean Feature Value')
    ax1.fill_between(range(len(mean_features)), 
                    mean_features - std_features, 
                    mean_features + std_features, 
                    alpha=0.3, label='±1 Std Dev')
    ax1.set_title('Feature Changes Over Time')
    ax1.set_ylabel('Feature Value')
    ax1.legend()
    ax1.grid(True)
    
    # Second subplot: Feature change rate
    ax2.plot(range(1, len(mean_features)), feature_diff, color='orange')
    ax2.set_title('Feature Change Rate (Potential Action Boundaries)')
    ax2.set_xlabel('Frame Index')
    ax2.set_ylabel('Change Rate')
    ax2.grid(True)
    
    # Mark large changes (potential action boundaries)
    threshold = np.std(feature_diff) * 2
    change_points = np.where(np.abs(feature_diff) > threshold)[0]
    ax2.scatter(change_points + 1, feature_diff[change_points], 
               color='red', s=50, label='Potential Action Boundary')
    
    if len(change_points) > 0:
        ax2.legend()
        
        # Also mark these points on the first plot
        for cp in change_points:
            ax1.axvline(x=cp+1, color='r', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
    plt.show()
    
    # Output potential action boundary positions
    if len(change_points) > 0:
        print("\nPotential Action Boundaries (Frame Index):")
        for i, cp in enumerate(sorted(change_points)):
            # Convert to time (assuming 24fps, 1 feature per 1.5 seconds)
            time_in_seconds = (cp + 1) * 1.5
            minutes = int(time_in_seconds // 60)
            seconds = time_in_seconds % 60
            print(f"  {i+1}. Frame {cp+1} (approx. {minutes:02d}:{seconds:05.2f})")

def compare_features(feature_files, output_path=None):
    """Compare multiple feature files"""
    if len(feature_files) < 2:
        print("Need at least two feature files for comparison")
        return
    
    features_list = []
    names = []
    
    # Load all features
    for file_path in feature_files:
        features = np.load(file_path)
        features_list.append(features)
        names.append(os.path.basename(file_path).split('.')[0])
    
    # Find common length
    min_length = min(f.shape[0] for f in features_list)
    
    # Truncate to common length
    features_list = [f[:min_length] for f in features_list]
    
    # Calculate similarity between features
    print("\nSimilarity between feature files (Cosine Similarity):")
    
    # Calculate mean feature vectors
    mean_features = [np.mean(f, axis=1) for f in features_list]
    
    # Calculate similarity matrix
    similarity_matrix = np.zeros((len(features_list), len(features_list)))
    for i in range(len(features_list)):
        for j in range(len(features_list)):
            # Calculate cosine similarity
            similarity = np.dot(mean_features[i], mean_features[j]) / (
                np.linalg.norm(mean_features[i]) * np.linalg.norm(mean_features[j]))
            similarity_matrix[i, j] = similarity
            
            if i < j:  # Only print upper triangle
                print(f"  {names[i]} vs {names[j]}: {similarity:.4f}")
    
    # Plot similarity matrix
    plt.figure(figsize=(10, 8))
    plt.imshow(similarity_matrix, cmap='viridis', vmin=0, vmax=1)
    plt.colorbar(label='Cosine Similarity')
    plt.title('Similarity Matrix Between Feature Files')
    plt.xticks(range(len(names)), names, rotation=45)
    plt.yticks(range(len(names)), names)
    
    for i in range(len(names)):
        for j in range(len(names)):
            plt.text(j, i, f"{similarity_matrix[i, j]:.2f}", 
                    ha="center", va="center", color="white" if similarity_matrix[i, j] < 0.7 else "black")
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
    plt.show()
    
    # Plot feature changes over time comparison
    plt.figure(figsize=(12, 6))
    
    for i, (features, name) in enumerate(zip(mean_features, names)):
        plt.plot(features, label=name, alpha=0.7)
    
    plt.title('Temporal Feature Comparison')
    plt.xlabel('Frame Index')
    plt.ylabel('Mean Feature Value')
    plt.legend()
    plt.grid(True)
    
    if output_path and output_path.endswith('.png'):
        comparison_path = output_path.replace('.png', '_temporal.png')
        plt.savefig(comparison_path)
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Feature File Viewer')
    parser.add_argument('files', nargs='+', help='Feature files to analyze (.npy)')
    parser.add_argument('--output', '-o', type=str, help='Output directory for images')
    parser.add_argument('--pca', type=int, choices=[2, 3], default=2,
                       help='PCA dimensions (2 or 3)')
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    if args.output:
        os.makedirs(args.output, exist_ok=True)
    
    # Process all files
    features_list = []
    for file_path in args.files:
        features = load_feature(file_path)
        if features is not None:
            features_list.append(features)
            
            # File name (without extension)
            file_name = os.path.basename(file_path).split('.')[0]
            
            # Output paths
            heatmap_path = os.path.join(args.output, f"{file_name}_heatmap.png") if args.output else None
            pca_path = os.path.join(args.output, f"{file_name}_pca.png") if args.output else None
            temporal_path = os.path.join(args.output, f"{file_name}_temporal.png") if args.output else None
            
            # Plot heatmap
            print("\nGenerating heatmap...")
            plot_feature_heatmap(features, heatmap_path)
            
            # Plot PCA reduction
            print("\nGenerating PCA visualization...")
            plot_feature_pca(features, args.pca, pca_path)
            
            # Plot temporal features
            print("\nGenerating temporal feature analysis...")
            plot_temporal_features(features, temporal_path)
    
    # If multiple files, compare them
    if len(features_list) > 1:
        comparison_path = os.path.join(args.output, "features_comparison.png") if args.output else None
        compare_features(args.files, comparison_path)

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
        print("If you encounter matplotlib errors, try:")
        print("  pip install matplotlib")
        print("  pip install scikit-learn") 