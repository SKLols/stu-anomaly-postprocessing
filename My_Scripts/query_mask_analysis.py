import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

def analyze_query_masks(mask_file_path, threshold=1e-6):
    """
    Analyze query mask statistics
    
    Args:
        mask_file_path: Path to the mask file [N_points, 100]
        threshold: Values below this are considered zero
    """
    # Load mask file
    masks = np.loadtxt(mask_file_path)  # [N_points, 100]
    print(f"Mask file: {mask_file_path}")
    print(f"Shape: {masks.shape} (points: {masks.shape[0]}, queries: {masks.shape[1]})\n")
    
    # Analysis for each query
    query_stats = []
    empty_queries = []
    
    for query_idx in range(masks.shape[1]):
        query_data = masks[:, query_idx]
        
        # Basic statistics
        non_zero_mask = query_data > threshold
        non_zero_count = np.sum(non_zero_mask)
        non_zero_values = query_data[non_zero_mask]
        
        stats = {
            'query_idx': query_idx,
            'total_points': len(query_data),
            'non_zero_points': non_zero_count,
            'zero_points': len(query_data) - non_zero_count,
            'zero_percentage': (len(query_data) - non_zero_count) / len(query_data) * 100,
            'min_value': np.min(query_data),
            'max_value': np.max(query_data),
            'mean_value': np.mean(query_data),
            'mean_non_zero': np.mean(non_zero_values) if len(non_zero_values) > 0 else 0,
            'unique_values': len(np.unique(query_data)),
            'is_empty': non_zero_count == 0
        }
        
        query_stats.append(stats)
        
        if stats['is_empty']:
            empty_queries.append(query_idx)
    
    # Print summary
    print("=== QUERY MASK ANALYSIS SUMMARY ===")
    print(f"Empty queries (all zeros): {len(empty_queries)}/100")
    print(f"Empty query indices: {empty_queries}")
    print(f"Non-empty queries: {100 - len(empty_queries)}/100\n")
    
    # Print detailed stats for first 20 queries
    print("=== DETAILED STATS (First 20 queries) ===")
    print("Query | Zero% | NonZero | Min    | Max    | Mean   | Unique | Empty")
    print("------|-------|---------|--------|--------|--------|--------|------")
    
    for stats in query_stats[:20]:
        print(f"{stats['query_idx']:5d} | "
              f"{stats['zero_percentage']:5.1f}% | "
              f"{stats['non_zero_points']:7d} | "
              f"{stats['min_value']:.4f} | "
              f"{stats['max_value']:.4f} | "
              f"{stats['mean_value']:.4f} | "
              f"{stats['unique_values']:6d} | "
              f"{'YES' if stats['is_empty'] else 'NO'}")
    
    # Find most and least active queries
    active_queries = [s for s in query_stats if not s['is_empty']]
    active_queries.sort(key=lambda x: x['non_zero_points'], reverse=True)
    
    print(f"\n=== TOP 10 MOST ACTIVE QUERIES ===")
    for i, stats in enumerate(active_queries[:10]):
        print(f"{i+1:2d}. Query {stats['query_idx']:3d}: "
              f"{stats['non_zero_points']:5d} points "
              f"({stats['non_zero_points']/stats['total_points']*100:5.1f}%) "
              f"[min:{stats['min_value']:.4f}, max:{stats['max_value']:.4f}]")
    
    return query_stats, empty_queries

def analyze_multiple_frames(mask_directory, num_frames=5):
    """
    Analyze multiple frames to see query consistency
    """
    mask_dir = Path(mask_directory)
    mask_files = sorted(mask_dir.glob("*_masks.txt"))[:num_frames]
    
    print("=== MULTI-FRAME QUERY CONSISTENCY ANALYSIS ===\n")
    
    frame_results = []
    for mask_file in mask_files:
        print(f"Analyzing: {mask_file.name}")
        query_stats, empty_queries = analyze_query_masks(mask_file)
        
        frame_results.append({
            'file': mask_file.name,
            'empty_queries': empty_queries,
            'non_empty_count': 100 - len(empty_queries),
            'top_queries': [s['query_idx'] for s in sorted(query_stats, 
                                                         key=lambda x: x['non_zero_points'], 
                                                         reverse=True)[:10]]
        })
        print("-" * 80)
    
    # Analyze consistency across frames
    print("\n=== QUERY CONSISTENCY ACROSS FRAMES ===")
    all_non_empty = []
    for result in frame_results:
        all_non_empty.extend([q for q in range(100) if q not in result['empty_queries']])
    
    from collections import Counter
    query_frequency = Counter(all_non_empty)
    
    print("Query appearance frequency across frames:")
    for query_idx, freq in query_frequency.most_common(20):
        print(f"Query {query_idx:3d}: appears in {freq}/{len(frame_results)} frames")

def plot_query_activity(mask_file_path):
    """
    Create visualization of query activity
    """
    masks = np.loadtxt(mask_file_path)
    
    # Calculate activity metrics
    non_zero_counts = []
    mean_activations = []
    
    for query_idx in range(masks.shape[1]):
        query_data = masks[:, query_idx]
        non_zero_count = np.sum(query_data > 1e-6)
        non_zero_counts.append(non_zero_count)
        mean_activations.append(np.mean(query_data))
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    
    # Plot 1: Non-zero points per query
    ax1.bar(range(100), non_zero_counts, color='skyblue', alpha=0.7)
    ax1.set_title('Number of Non-Zero Points per Query')
    ax1.set_xlabel('Query Index')
    ax1.set_ylabel('Non-Zero Points')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Mean activation per query
    ax2.bar(range(100), mean_activations, color='lightcoral', alpha=0.7)
    ax2.set_title('Mean Activation Value per Query')
    ax2.set_xlabel('Query Index')
    ax2.set_ylabel('Mean Activation')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('query_activity_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()

# Usage examples:
if __name__ == "__main__":
    # Analyze single frame
    mask_file = "Mask4Former3D/train_dataloader_dump/0_000000_masks.txt"
    query_stats, empty_queries = analyze_query_masks(mask_file)
    
    # Analyze multiple frames
    mask_directory = "/media/ubuntu22/HDD22T/_Sourabh/stu_dataset/Mask4Former3D/query_masks_dump/"
    analyze_multiple_frames(mask_directory, num_frames=3)
    
    # Create visualization
    plot_query_activity(mask_file)
