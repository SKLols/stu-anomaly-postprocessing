import numpy as np
from pathlib import Path

def reorder_queries_by_area(mask_file_path, output_suffix="_arranged"):
    """
    Reorder queries from smallest to largest area coverage
    """
    # Load mask file
    masks = np.loadtxt(mask_file_path)  # [N_points, 100]
    print(f"Original file: {mask_file_path}")
    print(f"Shape: {masks.shape}\n")
    
    # Calculate area coverage for each query
    query_areas = []
    threshold = 1e-6
    
    for query_idx in range(masks.shape[1]):
        query_data = masks[:, query_idx]
        non_zero_count = np.sum(query_data > threshold)
        query_areas.append((query_idx, non_zero_count))
    
    # Sort by area coverage (smallest first)
    query_areas.sort(key=lambda x: x[1])
    
    # Create new ordered mask array
    ordered_masks = np.zeros_like(masks)
    new_order = []
    
    for new_idx, (original_idx, area) in enumerate(query_areas):
        ordered_masks[:, new_idx] = masks[:, original_idx]
        new_order.append((new_idx, original_idx, area))
    
    # Save reordered file
    output_path = mask_file_path.replace('.txt', f'{output_suffix}.txt')
    np.savetxt(output_path, ordered_masks, fmt='%.6f')
    
    # Print reordering summary
    print("=== QUERY REORDERING SUMMARY ===")
    print("NewIdx | OriginalIdx | AreaCoverage | Coverage%")
    print("-------|-------------|--------------|----------")
    
    for new_idx, original_idx, area in new_order:
        coverage_pct = (area / masks.shape[0]) * 100
        print(f"{new_idx:6d} | {original_idx:11d} | {area:12d} | {coverage_pct:8.2f}%")
    
    print(f"\n✅ Saved reordered queries to: {output_path}")
    
    return ordered_masks, new_order

def analyze_reordered_queries(mask_file_path):
    """
    Analyze both original and reordered queries
    """
    # Reorder queries
    ordered_masks, new_order = reorder_queries_by_area(mask_file_path)
    
    # Analyze the reordered result
    print(f"\n=== REORDERED QUERY ANALYSIS ===")
    
    # Group by coverage size
    small_queries = [q for q in new_order if q[2] < 100]  # < 100 points
    medium_queries = [q for q in new_order if 100 <= q[2] < 1000]  # 100-1k points
    large_queries = [q for q in new_order if 1000 <= q[2] < 10000]  # 1k-10k points
    huge_queries = [q for q in new_order if q[2] >= 10000]  # 10k+ points
    
    print(f"Small queries (<100 points): {len(small_queries)}")
    print(f"Medium queries (100-1k points): {len(medium_queries)}")
    print(f"Large queries (1k-10k points): {len(large_queries)}")
    print(f"Huge queries (10k+ points): {len(huge_queries)}")
    
    # Show examples from each category
    print(f"\n=== QUERY CATEGORIES ===")
    
    if small_queries:
        print(f"\nSMALL QUERIES (focused detectors):")
        for new_idx, orig_idx, area in small_queries[:5]:  # First 5
            print(f"  New {new_idx:2d} (Orig {orig_idx:2d}): {area:4d} points")
    
    if medium_queries:
        print(f"\nMEDIUM QUERIES (object-level):")
        for new_idx, orig_idx, area in medium_queries[:5]:
            print(f"  New {new_idx:2d} (Orig {orig_idx:2d}): {area:4d} points")
    
    if large_queries:
        print(f"\nLARGE QUERIES (region-level):")
        for new_idx, orig_idx, area in large_queries[:5]:
            print(f"  New {new_idx:2d} (Orig {orig_idx:2d}): {area:5d} points")
    
    if huge_queries:
        print(f"\nHUGE QUERIES (scene-level):")
        for new_idx, orig_idx, area in huge_queries:
            coverage_pct = (area / ordered_masks.shape[0]) * 100
            print(f"  New {new_idx:2d} (Orig {orig_idx:2d}): {area:6d} points ({coverage_pct:5.1f}%)")

def batch_reorder_masks(mask_directory, output_suffix="_arranged"):
    """
    Reorder all mask files in a directory
    """
    mask_dir = Path(mask_directory)
    mask_files = sorted(mask_dir.glob("*_masks.txt"))
    
    print(f"=== BATCH REORDERING {len(mask_files)} FILES ===")
    
    for mask_file in mask_files:
        print(f"\nProcessing: {mask_file.name}")
        try:
            reorder_queries_by_area(str(mask_file), output_suffix)
        except Exception as e:
            print(f"Error processing {mask_file}: {e}")
    
    print(f"\n🎉 Batch reordering complete!")

# Usage examples:
if __name__ == "__main__":
    # Single file reordering
    mask_file = "/media/ubuntu22/HDD22T/_Sourabh/stu_dataset/Mask4Former3D/query_masks_dump/125_000000_masks.txt"
    analyze_reordered_queries(mask_file)
    
    # Batch reordering
    mask_directory = "/media/ubuntu22/HDD22T/_Sourabh/stu_dataset/Mask4Former3D/query_masks_dump/"
    batch_reorder_masks(mask_directory)