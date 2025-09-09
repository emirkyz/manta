from sklearn.manifold import TSNE
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path
from typing import Optional, Union, List

def tsne_graph_output(w: np.ndarray, h: np.ndarray, tdm, vocab: List[str], 
                      output_dir: Optional[Union[str, Path]] = None, 
                      table_name: str = "tsne_plot", 
                      n_topic_words: int = 3,
                      time_data: Optional[pd.Series] = None,
                      time_ranges: Optional[List] = None,
                      cumulative: bool = True,
                      time_column_name: str = "time") -> Optional[str]:
    # Input validation and debugging
    print(f"t-SNE Debug Info:")
    print(f"- W shape: {w.shape if w is not None else 'None'}")
    print(f"- H shape: {h.shape if h is not None else 'None'}")
    print(f"- Vocab length: {len(vocab) if vocab is not None else 'None'}")
    print(f"- Time data: {'Yes' if time_data is not None else 'No'}")
    print(f"- Time ranges: {time_ranges}")
    print(f"- W type: {type(w)}")
    print(f"- H type: {type(h)}")
    
    if w is None or h is None or vocab is None or len(vocab) == 0:
        print("Warning: Invalid input data for t-SNE visualization")
        return None
    
    if w.shape[0] < 2:
        print("Warning: Not enough documents for t-SNE visualization")
        return None
    
    print(f"Generating t-SNE visualization for {w.shape[0]} documents and {h.shape[0]} topics...")
    
    # Convert W to dense array only if necessary
    if hasattr(w, 'toarray'):
        # It's a sparse matrix, convert to dense
        w_dense = w.toarray()
        print(f"Converted sparse matrix to dense: {w.shape} -> {w_dense.shape}")
    else:
        # Use np.asarray to avoid copying if already an array
        w_dense = np.asarray(w)
        print(f"Using matrix as-is: {w_dense.shape}")
    
    # Apply t-SNE to document-topic matrix (W) with optimized parameters
    n_docs = w_dense.shape[0]
    
    # Adaptive perplexity based on dataset size
    adaptive_perplexity = min(30, max(5, n_docs // 3))
    
    # Choose method based on dataset size
    method = 'barnes_hut' if n_docs > 1000 else 'exact'
    
    print(f"t-SNE parameters: perplexity={adaptive_perplexity}, method={method}, n_iter=300")
    
    tsne = TSNE(
        random_state=3211, 
        perplexity=adaptive_perplexity,
        n_iter=300,  # Reduced from default 1000
        learning_rate='auto',
        method=method
    )
    tsne_embedding = tsne.fit_transform(w_dense)
    tsne_embedding = pd.DataFrame(tsne_embedding, columns=['x', 'y'])
    tsne_embedding['hue'] = w_dense.argmax(axis=1)
    
    # Use all points for visualization (removed representative sampling)

    # Create meaningful topic labels from top words
    topics = []
    for i in range(h.shape[0]):
        try:
            # Handle both dense and sparse matrices
            if hasattr(h[i], 'toarray'):
                # Sparse matrix
                h_topic = h[i].toarray().flatten()
            elif hasattr(h[i], 'flatten'):
                # Dense array
                h_topic = h[i].flatten()
            else:
                # Already 1D
                h_topic = h[i]
            
            # Ensure we don't exceed vocabulary bounds
            max_vocab_idx = min(len(h_topic), len(vocab))
            if max_vocab_idx == 0:
                topics.append(f"Topic {i}: [no valid words]")
                continue
                
            # Get top word indices from valid range
            topic_scores = h_topic[:max_vocab_idx]
            top_indices = np.argsort(topic_scores)[-min(n_topic_words, len(topic_scores)):][::-1]
            
            # Get actual words, with double bounds checking
            top_words = []
            for idx in top_indices:
                if 0 <= idx < len(vocab) and 0 <= idx < len(topic_scores):
                    top_words.append(vocab[idx])
            
            if top_words:
                topic_label = f"Topic {i}: {', '.join(top_words)}"
            else:
                topic_label = f"Topic {i}: [no words found]"
            topics.append(topic_label)
            
        except Exception as e:
            print(f"Warning: Error processing topic {i}: {e}")
            topics.append(f"Topic {i}: [error]")
    
    legend_list = []

    # Check if time-series visualization is requested
    if time_data is not None and time_ranges is not None:
        return _create_time_series_visualization(
            tsne_embedding, topics, time_data, time_ranges, cumulative,
            output_dir, table_name, time_column_name
        )

    # Create the standard single visualization
    plt.style.use('ggplot')
    fig, ax = plt.subplots(1, 1, figsize=(12, 8), facecolor='w', edgecolor='k')
    
    data = tsne_embedding
    
    # Calculate adaptive point size and alpha based on dataset size
    n_points = len(data)
    point_size = max(5, 20 - np.log10(max(n_points, 10)) * 3)  # Smaller points for larger datasets
    alpha_value = max(0.3, 1.0 - (n_points / 5000))  # More transparent for larger datasets
    
    print(f"Visualization settings: point_size={point_size:.1f}, alpha={alpha_value:.2f}")
    
    scatter = ax.scatter(data=data, x='x', y='y', s=point_size, c=data['hue'], 
                        cmap="Set1", alpha=alpha_value, edgecolors='black', linewidth=0.2)
    
    # Set title and labels
    ax.set_title(f't-SNE Visualization of Document Topics\n{table_name}', 
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
    
    # Create custom legend with simpler color mapping
    import matplotlib.cm as cm
    unique_topics = sorted(data['hue'].unique())
    
    # Use the same colormap as the scatter plot
    cmap = cm.get_cmap('Set1')
    
    for i, topic_id in enumerate(unique_topics):
        if topic_id < len(topics):  # Ensure we don't exceed topics list
            # Get color from colormap and convert to tuple
            color_normalized = topic_id / max(len(unique_topics) - 1, 1) if len(unique_topics) > 1 else 0
            color = cmap(color_normalized)
            # Convert numpy array to tuple if needed
            if hasattr(color, 'tolist'):
                color = tuple(color.tolist())
            legend_list.append(mpatches.Patch(color=color, label=topics[topic_id]))
        else:
            print(f"Warning: Topic ID {topic_id} exceeds available topics ({len(topics)})")
    
    # Add legend with proper positioning
    ax.legend(handles=legend_list, loc='center left', bbox_to_anchor=(1, 0.5), 
             fontsize=10, frameon=True, fancybox=True, shadow=True)
    
    plt.tight_layout()
    
    # Save the plot if output directory is provided
    saved_path = None
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        filename = f"{table_name}_tsne_visualization.png"
        file_path = output_path / filename
        
        plt.savefig(file_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        saved_path = str(file_path)
        print(f"t-SNE plot saved to: {saved_path}")
    
    plt.show()
    
    # Print summary statistics
    print(f"\nt-SNE Visualization Summary:")
    print(f"- Total documents: {len(data)}")
    print(f"- Number of topics: {len(unique_topics)}")
    for i, topic in enumerate(topics[:len(unique_topics)]):
        topic_count = len(data[data['hue'] == i])
        print(f"- {topic}: {topic_count} documents")
    
    return saved_path


def _create_time_series_visualization(tsne_embedding: pd.DataFrame, topics: List[str], 
                                     time_data: pd.Series, time_ranges: List, 
                                     cumulative: bool, output_dir: Optional[Union[str, Path]], 
                                     table_name: str, time_column_name: str) -> Optional[str]:
    """
    Create time-series t-SNE visualization with multiple subplots showing evolution over time.
    
    Args:
        tsne_embedding: DataFrame with x, y coordinates and hue (topic assignments)
        topics: List of topic labels
        time_data: Series containing time/date information for each document
        time_ranges: List of time points to create subplots for
        cumulative: If True, show "up to time X", if False show "only time X"
        output_dir: Directory to save the plot
        table_name: Base name for output files
        time_column_name: Name of the time column for display
        
    Returns:
        Path to saved plot or None if failed
    """
    import matplotlib.cm as cm
    from datetime import datetime
    
    print(f"Creating time-series t-SNE visualization with {len(time_ranges)} time periods...")
    
    # Parse time data if needed
    parsed_time_data = _parse_time_data(time_data)
    if parsed_time_data is None:
        print("Warning: Could not parse time data, falling back to standard visualization")
        return None
    
    # Create subplot layout (2x3 for 6 periods, 3x2 for 6 periods, etc.)
    n_periods = len(time_ranges)
    if n_periods <= 4:
        rows, cols = 2, 2
    elif n_periods <= 6:
        rows, cols = 2, 3
    elif n_periods <= 9:
        rows, cols = 3, 3
    else:
        rows, cols = 3, 4  # Max 12 periods
    
    # Create figure and subplots
    plt.style.use('ggplot')
    fig, axes = plt.subplots(rows, cols, figsize=(cols*5, rows*4), facecolor='w', edgecolor='k')
    fig.subplots_adjust(hspace=0.3, wspace=0.2)
    
    if n_periods == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    # Use consistent colormap
    cmap = cm.get_cmap('Set1')
    unique_topics = sorted(tsne_embedding['hue'].unique())
    
    # Create legend items (will be used for the main figure legend)
    legend_list = []
    for topic_id in unique_topics:
        if topic_id < len(topics):
            color_normalized = topic_id / max(len(unique_topics) - 1, 1) if len(unique_topics) > 1 else 0
            color = cmap(color_normalized)
            if hasattr(color, 'tolist'):
                color = tuple(color.tolist())
            legend_list.append(mpatches.Patch(color=color, label=topics[topic_id]))
    
    # Create each time period subplot
    for idx, time_point in enumerate(time_ranges[:n_periods]):  # Limit to available subplots
        ax = axes[idx]
        
        # Filter data based on time
        if cumulative:
            # Show all data up to this time point
            mask = parsed_time_data <= time_point
            title_prefix = f"Until {time_point}"
        else:
            # Show only data from this specific time period (you might want to define ranges)
            mask = parsed_time_data == time_point
            title_prefix = f"In {time_point}"
        
        filtered_data = tsne_embedding[mask]
        
        if len(filtered_data) > 0:
            # Calculate adaptive visualization settings for this subplot
            n_subplot_points = len(filtered_data)
            subplot_point_size = max(4, 18 - np.log10(max(n_subplot_points, 10)) * 3)
            subplot_alpha = max(0.4, 1.0 - (n_subplot_points / 3000))
            
            # Create scatter plot
            scatter = ax.scatter(
                data=filtered_data, x='x', y='y', s=subplot_point_size, c=filtered_data['hue'], 
                cmap="Set1", alpha=subplot_alpha, edgecolors='black', linewidth=0.15
            )
            
            ax.set_title(f'{title_prefix}\\n({len(filtered_data)} documents)', 
                        fontsize=10, fontweight='bold')
        else:
            ax.set_title(f'{title_prefix}\\n(No data)', fontsize=10, color='gray')
        
        ax.axis('off')
    
    # Hide unused subplots
    for idx in range(n_periods, len(axes)):
        axes[idx].axis('off')
    
    # Add main title and legend
    title_type = "Cumulative" if cumulative else "Period-by-Period"
    fig.suptitle(f't-SNE Topic Evolution Over Time ({title_type})\\n{table_name}', 
                fontsize=16, fontweight='bold', y=0.95)
    
    # Add legend to the figure
    fig.legend(legend_list, loc='lower center', bbox_to_anchor=(0.5, -0.05), 
              ncol=min(3, len(legend_list)), fontsize=10, frameon=True, 
              fancybox=True, shadow=True)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)  # Make room for legend
    
    # Save the plot
    saved_path = None
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        filename = f"{table_name}_tsne_time_series.png"
        file_path = output_path / filename
        
        plt.savefig(file_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        saved_path = str(file_path)
        print(f"Time-series t-SNE plot saved to: {saved_path}")
    
    plt.show()
    
    # Print summary
    print(f"\\nTime-Series t-SNE Summary:")
    print(f"- Time periods: {len(time_ranges)}")
    print(f"- Mode: {'Cumulative' if cumulative else 'Period-by-period'}")
    for i, time_point in enumerate(time_ranges):
        if cumulative:
            mask = parsed_time_data <= time_point
        else:
            mask = parsed_time_data == time_point
        count = mask.sum()
        print(f"- {time_point}: {count} documents")
    
    return saved_path


def _parse_time_data(time_data: pd.Series) -> Optional[pd.Series]:
    """
    Parse various time formats into a standardized format for filtering.
    
    Args:
        time_data: Series containing time/date information
        
    Returns:
        Parsed time series or None if parsing fails
    """
    import pandas as pd
    from datetime import datetime
    
    if time_data is None or len(time_data) == 0:
        return None
    
    try:
        # Try different parsing strategies
        
        # Strategy 1: Already datetime
        if pd.api.types.is_datetime64_any_dtype(time_data):
            return time_data
        
        # Strategy 2: Numeric years (2020, 2021, etc.)
        if pd.api.types.is_numeric_dtype(time_data):
            # Assume years if values are reasonable (1900-2100)
            min_val, max_val = time_data.min(), time_data.max()
            if 1900 <= min_val <= 2100 and 1900 <= max_val <= 2100:
                return pd.to_datetime(time_data, format='%Y', errors='coerce')
        
        # Strategy 3: String parsing
        if pd.api.types.is_string_dtype(time_data):
            # Try common formats
            for fmt in ['%Y', '%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y', '%Y-%m-%d %H:%M:%S']:
                try:
                    parsed = pd.to_datetime(time_data, format=fmt, errors='coerce')
                    if not parsed.isna().all():
                        return parsed
                except:
                    continue
            
            # Try general parsing
            try:
                return pd.to_datetime(time_data, errors='coerce')
            except:
                pass
        
        print(f"Warning: Could not parse time data of type {time_data.dtype}")
        return None
        
    except Exception as e:
        print(f"Warning: Error parsing time data: {e}")
        return None


def _auto_detect_time_ranges(time_data: pd.Series, n_periods: int = 6) -> Optional[List]:
    """
    Automatically detect meaningful time ranges from the data.
    
    Args:
        time_data: Parsed time series
        n_periods: Number of time periods to create
        
    Returns:
        List of time points for subplots
    """
    if time_data is None or len(time_data) == 0:
        return None
    
    try:
        # Remove NaN values
        valid_times = time_data.dropna()
        if len(valid_times) == 0:
            return None
        
        min_time = valid_times.min()
        max_time = valid_times.max()
        
        # Create evenly spaced time points
        time_range = pd.date_range(start=min_time, end=max_time, periods=n_periods)
        
        # Convert to year if the span is multiple years
        time_span_years = (max_time - min_time).days / 365.25
        if time_span_years > 2:
            # Use years
            return [t.year for t in time_range]
        else:
            # Use full dates
            return [t.strftime('%Y-%m-%d') for t in time_range]
    
    except Exception as e:
        print(f"Warning: Could not auto-detect time ranges: {e}")
        return None


def _apply_representative_sampling(tsne_data: pd.DataFrame, target_size: int = 1500) -> pd.DataFrame:
    """
    Apply representative sampling to reduce point density while preserving data distribution.
    
    Args:
        tsne_data: DataFrame with x, y, hue columns
        target_size: Target number of points to keep
        
    Returns:
        Sampled DataFrame with representative points
    """
    from sklearn.cluster import KMeans
    import numpy as np
    
    if len(tsne_data) <= target_size:
        return tsne_data
    
    try:
        # Separate by topic to maintain topic representation
        sampled_dfs = []
        
        for hue in tsne_data['hue'].unique():
            topic_data = tsne_data[tsne_data['hue'] == hue].copy()
            topic_size = len(topic_data)
            
            # Proportional sampling: each topic gets proportional representation
            topic_target = max(10, int(target_size * (topic_size / len(tsne_data))))
            topic_target = min(topic_target, topic_size)
            
            if topic_size <= topic_target:
                # Keep all points for small topics
                sampled_dfs.append(topic_data)
            else:
                # Apply clustering-based sampling for large topics
                coords = topic_data[['x', 'y']].values
                
                # Use fewer clusters for better performance
                n_clusters = min(topic_target // 2, topic_size // 4, 50)
                
                if n_clusters < 2:
                    # Random sampling for very small targets
                    sampled = topic_data.sample(n=topic_target, random_state=42)
                else:
                    # Cluster-based sampling
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                    clusters = kmeans.fit_predict(coords)
                    
                    sampled_indices = []
                    
                    # From each cluster, take cluster center + some random points
                    for cluster_id in range(n_clusters):
                        cluster_mask = clusters == cluster_id
                        cluster_indices = topic_data.index[cluster_mask]
                        
                        if len(cluster_indices) == 0:
                            continue
                        
                        # Take 1-3 points per cluster depending on cluster size
                        points_per_cluster = max(1, min(3, topic_target // n_clusters))
                        
                        if len(cluster_indices) <= points_per_cluster:
                            sampled_indices.extend(cluster_indices)
                        else:
                            # Take the point closest to cluster center + random samples
                            cluster_coords = coords[cluster_mask]
                            center = kmeans.cluster_centers_[cluster_id]
                            
                            # Find closest point to center
                            distances = np.sum((cluster_coords - center) ** 2, axis=1)
                            closest_idx = cluster_indices[np.argmin(distances)]
                            sampled_indices.append(closest_idx)
                            
                            # Add random samples from the cluster
                            remaining_indices = [idx for idx in cluster_indices if idx != closest_idx]
                            if remaining_indices and points_per_cluster > 1:
                                n_random = min(points_per_cluster - 1, len(remaining_indices))
                                random_indices = np.random.choice(remaining_indices, n_random, replace=False)
                                sampled_indices.extend(random_indices)
                    
                    # If we have fewer points than target, add some random ones
                    if len(sampled_indices) < topic_target:
                        remaining_indices = [idx for idx in topic_data.index if idx not in sampled_indices]
                        if remaining_indices:
                            n_additional = min(topic_target - len(sampled_indices), len(remaining_indices))
                            additional_indices = np.random.choice(remaining_indices, n_additional, replace=False)
                            sampled_indices.extend(additional_indices)
                    
                    sampled = topic_data.loc[sampled_indices]
                
                sampled_dfs.append(sampled)
        
        # Combine all sampled topics
        result = pd.concat(sampled_dfs, ignore_index=True)
        return result
        
    except Exception as e:
        print(f"Warning: Representative sampling failed: {e}")
        # Fallback to simple random sampling
        return tsne_data.sample(n=min(target_size, len(tsne_data)), random_state=42)