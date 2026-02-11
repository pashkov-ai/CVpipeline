"""
Comprehensive EDA module for AITEX Fabric Dataset.

This module combines statistical analysis and visualization functions
for exploratory data analysis of fabric defect detection data.
"""

import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from typing import Optional

from data_loader import load_mask, get_mask_bbox, load_image_triplet, get_sample_images

# Set visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.facecolor'] = 'white'


# ============================================================================
# STATISTICAL ANALYSIS FUNCTIONS
# ============================================================================

def compute_mask_statistics(mask: np.ndarray, image_height: int,
                            image_width: int) -> dict:
    """
    Compute statistics from a binary mask.

    Args:
        mask: Binary mask image
        image_height: Height of original image
        image_width: Width of original image

    Returns:
        Dictionary with defect statistics
    """
    bbox = get_mask_bbox(mask)

    if bbox is None:
        return {
            'defect_area': 0,
            'defect_pixels': 0,
            'defect_percentage': 0.0,
            'bbox_x': 0,
            'bbox_y': 0,
            'bbox_width': 0,
            'bbox_height': 0,
            'aspect_ratio': 0.0,
            'center_x': 0.0,
            'center_y': 0.0,
            'center_x_norm': 0.0,
            'center_y_norm': 0.0
        }

    # Count defect pixels
    defect_pixels = np.sum(mask > 0)
    total_pixels = image_height * image_width

    # Calculate center of defect
    center_x = bbox['x'] + bbox['width'] / 2
    center_y = bbox['y'] + bbox['height'] / 2

    # Aspect ratio
    aspect_ratio = bbox['width'] / bbox['height'] if bbox['height'] > 0 else 0

    return {
        'defect_area': bbox['width'] * bbox['height'],
        'defect_pixels': int(defect_pixels),
        'defect_percentage': (defect_pixels / total_pixels) * 100,
        'bbox_x': bbox['x'],
        'bbox_y': bbox['y'],
        'bbox_width': bbox['width'],
        'bbox_height': bbox['height'],
        'aspect_ratio': aspect_ratio,
        'center_x': center_x,
        'center_y': center_y,
        'center_x_norm': center_x / image_width,  # Normalized to [0, 1]
        'center_y_norm': center_y / image_height
    }


def add_mask_statistics_to_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add mask statistics to dataset DataFrame.

    Args:
        df: Dataset DataFrame with mask_path column

    Returns:
        DataFrame with added statistics columns
    """
    df = df.copy()

    # Initialize columns
    stat_cols = [
        'defect_area', 'defect_pixels', 'defect_percentage',
        'bbox_x', 'bbox_y', 'bbox_width', 'bbox_height',
        'aspect_ratio', 'center_x', 'center_y',
        'center_x_norm', 'center_y_norm'
    ]

    for col in stat_cols:
        df[col] = np.nan

    # Process defect images only
    defect_mask = df['has_defect']

    for idx in df[defect_mask].index:
        mask_path = df.loc[idx, 'mask_path']

        if mask_path and pd.notna(mask_path):
            try:
                mask = load_mask(mask_path)
                stats = compute_mask_statistics(mask, mask.shape[0], mask.shape[1])

                for key, value in stats.items():
                    df.loc[idx, key] = value
            except Exception as e:
                print(f"Error processing mask {mask_path}: {e}")

    return df


def compute_defect_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute statistics per defect type.

    Args:
        df: Dataset DataFrame with mask statistics

    Returns:
        DataFrame with defect type statistics
    """
    defect_df = df[df['has_defect']].copy()

    stats = defect_df.groupby(['defect_code', 'defect_name']).agg({
        'image_id': 'count',
        'defect_area': ['mean', 'std', 'min', 'max', 'median'],
        'defect_percentage': ['mean', 'std', 'min', 'max'],
        'aspect_ratio': ['mean', 'median'],
        'bbox_width': ['mean', 'median'],
        'bbox_height': ['mean', 'median']
    }).reset_index()

    # Flatten column names
    stats.columns = ['_'.join(col).strip('_') if col[1] else col[0]
                     for col in stats.columns.values]

    # Rename for clarity
    stats = stats.rename(columns={'image_id_count': 'count'})

    # Add percentage
    stats['percentage'] = (stats['count'] / len(defect_df)) * 100

    # Round numeric columns
    numeric_cols = stats.select_dtypes(include=[np.number]).columns
    stats[numeric_cols] = stats[numeric_cols].round(2)

    return stats.sort_values('count', ascending=False)


def compute_fabric_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute statistics per fabric type.

    Args:
        df: Dataset DataFrame

    Returns:
        DataFrame with fabric type statistics
    """
    stats = df.groupby('fabric_code').agg({
        'image_id': 'count',
        'has_defect': 'sum'
    }).reset_index()

    stats.columns = ['fabric_code', 'total_images', 'defect_images']
    stats['no_defect_images'] = stats['total_images'] - stats['defect_images']
    stats['defect_rate'] = (stats['defect_images'] / stats['total_images']) * 100

    stats = stats.round(2)
    return stats.sort_values('fabric_code')


def compute_fabric_defect_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create cross-tabulation of fabric types vs defect types.

    Args:
        df: Dataset DataFrame

    Returns:
        Pivot table with fabric codes as rows and defect codes as columns
    """
    defect_df = df[df['has_defect']]

    matrix = pd.crosstab(
        defect_df['fabric_code'],
        defect_df['defect_name'],
        margins=False
    )

    return matrix


def compute_spatial_statistics(df: pd.DataFrame) -> dict:
    """
    Analyze spatial distribution of defects.

    Args:
        df: Dataset DataFrame with center coordinates

    Returns:
        Dictionary with spatial statistics
    """
    defect_df = df[df['has_defect']].copy()

    # Define regions
    def get_horizontal_region(x_norm):
        if x_norm < 0.33:
            return 'left'
        elif x_norm < 0.67:
            return 'center'
        else:
            return 'right'

    def get_vertical_region(y_norm):
        if y_norm < 0.33:
            return 'top'
        elif y_norm < 0.67:
            return 'middle'
        else:
            return 'bottom'

    defect_df['h_region'] = defect_df['center_x_norm'].apply(get_horizontal_region)
    defect_df['v_region'] = defect_df['center_y_norm'].apply(get_vertical_region)

    h_dist = defect_df['h_region'].value_counts().to_dict()
    v_dist = defect_df['v_region'].value_counts().to_dict()

    return {
        'horizontal_distribution': h_dist,
        'vertical_distribution': v_dist,
        'mean_x_norm': defect_df['center_x_norm'].mean(),
        'mean_y_norm': defect_df['center_y_norm'].mean(),
        'std_x_norm': defect_df['center_x_norm'].std(),
        'std_y_norm': defect_df['center_y_norm'].std()
    }


def get_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create overall dataset summary table.

    Args:
        df: Dataset DataFrame

    Returns:
        Summary DataFrame
    """
    summary_data = []

    # Overall stats
    summary_data.append({
        'Metric': 'Total Images',
        'Value': len(df)
    })

    summary_data.append({
        'Metric': 'Defect Images',
        'Value': df['has_defect'].sum()
    })

    summary_data.append({
        'Metric': 'Non-Defect Images',
        'Value': (~df['has_defect']).sum()
    })

    summary_data.append({
        'Metric': 'Defect Types',
        'Value': df[df['has_defect']]['defect_code'].nunique()
    })

    summary_data.append({
        'Metric': 'Fabric Types',
        'Value': df['fabric_code'].nunique()
    })

    # Image dimensions (assuming standard size)
    if len(df) > 0 and 'image_path' in df.columns:
        try:
            sample_img = cv2.imread(df.iloc[0]['image_path'])
            if sample_img is not None:
                h, w = sample_img.shape[:2]
                summary_data.append({
                    'Metric': 'Image Resolution',
                    'Value': f'{w} × {h}'
                })
        except Exception:
            pass

    # Defect statistics
    if df['has_defect'].any():
        defect_df = df[df['has_defect']]

        if 'defect_area' in defect_df.columns:
            summary_data.append({
                'Metric': 'Avg Defect Area (pixels)',
                'Value': f"{defect_df['defect_area'].mean():.0f}"
            })

        if 'defect_percentage' in defect_df.columns:
            summary_data.append({
                'Metric': 'Avg Defect Coverage (%)',
                'Value': f"{defect_df['defect_percentage'].mean():.3f}"
            })

    return pd.DataFrame(summary_data)


def analyze_defect_orientation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze defect orientation (horizontal vs vertical).

    Args:
        df: Dataset DataFrame with aspect_ratio

    Returns:
        DataFrame with orientation analysis
    """
    defect_df = df[df['has_defect']].copy()

    def get_orientation(aspect_ratio):
        if pd.isna(aspect_ratio) or aspect_ratio == 0:
            return 'unknown'
        elif aspect_ratio > 2:
            return 'horizontal'
        elif aspect_ratio < 0.5:
            return 'vertical'
        else:
            return 'square'

    defect_df['orientation'] = defect_df['aspect_ratio'].apply(get_orientation)

    orientation_stats = defect_df.groupby('orientation').agg({
        'image_id': 'count',
        'aspect_ratio': ['mean', 'median', 'min', 'max']
    }).reset_index()

    orientation_stats.columns = ['_'.join(col).strip('_') if col[1] else col[0]
                                  for col in orientation_stats.columns.values]

    return orientation_stats


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_image_triplet(image_path: str, mask_path: str,
                       defect_name: str = "",
                       figsize: tuple[int, int] = (15, 4),
                       save_path: Optional[str] = None):
    """
    Plot side-by-side: original image, mask, and cropped defect.

    Args:
        image_path: Path to original image
        mask_path: Path to mask image
        defect_name: Name of defect for title
        figsize: Figure size
        save_path: Optional path to save figure
    """
    try:
        original, mask, cropped = load_image_triplet(image_path, mask_path, padding=20)

        fig, axes = plt.subplots(1, 3, figsize=figsize)

        # Original image
        axes[0].imshow(original)
        axes[0].set_title('Original Image', fontsize=12, fontweight='bold')
        axes[0].axis('off')

        # Mask
        axes[1].imshow(mask, cmap='gray')
        axes[1].set_title('Binary Mask', fontsize=12, fontweight='bold')
        axes[1].axis('off')

        # Cropped defect
        if cropped is not None and cropped.size > 0:
            axes[2].imshow(cropped)
            axes[2].set_title('Cropped Defect', fontsize=12, fontweight='bold')
            axes[2].axis('off')
        else:
            axes[2].text(0.5, 0.5, 'No defect found',
                        ha='center', va='center', fontsize=12)
            axes[2].axis('off')

        if defect_name:
            fig.suptitle(f'Defect: {defect_name}', fontsize=14, fontweight='bold', y=1.02)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')

        return fig

    except Exception as e:
        print(f"Error plotting triplet: {e}")
        return None


def plot_multiple_triplets(df: pd.DataFrame, n_samples: int = 12,
                           cols: int = 3, figsize: tuple[int, int] = (18, 20),
                           save_path: Optional[str] = None):
    """
    Plot multiple image triplets in a grid.

    Args:
        df: DataFrame with sample images
        n_samples: Number of samples to plot
        cols: Number of columns in grid
        figsize: Figure size
        save_path: Optional path to save figure
    """
    samples = df.head(n_samples)
    rows = int(np.ceil(len(samples) / cols))

    fig = plt.figure(figsize=figsize)

    for idx, (_, row) in enumerate(samples.iterrows()):
        try:
            original, mask, cropped = load_image_triplet(
                row['image_path'],
                row['mask_path'],
                padding=20
            )

            # Original
            ax1 = plt.subplot(rows, cols * 3, idx * 3 + 1)
            ax1.imshow(original)
            if idx < cols:
                ax1.set_title('Original', fontsize=10, fontweight='bold')
            ax1.axis('off')

            # Mask
            ax2 = plt.subplot(rows, cols * 3, idx * 3 + 2)
            ax2.imshow(mask, cmap='gray')
            if idx < cols:
                ax2.set_title('Mask', fontsize=10, fontweight='bold')
            ax2.axis('off')

            # Cropped
            ax3 = plt.subplot(rows, cols * 3, idx * 3 + 3)
            if cropped is not None and cropped.size > 0:
                ax3.imshow(cropped)
            else:
                ax3.text(0.5, 0.5, 'Empty', ha='center', va='center')
            if idx < cols:
                ax3.set_title('Cropped', fontsize=10, fontweight='bold')
            ax3.axis('off')

            # Add defect name on the left
            if idx % 3 == 0:
                ax1.text(-0.1, 0.5, row['defect_name'],
                        transform=ax1.transAxes,
                        rotation=90, va='center', ha='right',
                        fontsize=9, fontweight='bold')

        except Exception as e:
            print(f"Error plotting row {idx}: {e}")
            continue

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')

    return fig


def plot_defect_distribution(df: pd.DataFrame, figsize: tuple[int, int] = (12, 6)):
    """
    Plot defect type distribution as bar chart.

    Args:
        df: Dataset DataFrame
        figsize: Figure size
    """
    defect_df = df[df['has_defect']]

    defect_counts = defect_df['defect_name'].value_counts().sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=figsize)

    colors = sns.color_palette("husl", len(defect_counts))
    defect_counts.plot(kind='barh', ax=ax, color=colors)

    ax.set_xlabel('Count', fontsize=12, fontweight='bold')
    ax.set_ylabel('Defect Type', fontsize=12, fontweight='bold')
    ax.set_title('Distribution of Defect Types', fontsize=14, fontweight='bold')

    # Add count labels
    for i, (idx, val) in enumerate(defect_counts.items()):
        ax.text(val + 0.5, i, str(val), va='center', fontsize=10)

    plt.tight_layout()
    return fig


def plot_fabric_distribution(df: pd.DataFrame, figsize: tuple[int, int] = (10, 6)):
    """
    Plot fabric type distribution.

    Args:
        df: Dataset DataFrame
        figsize: Figure size
    """
    fabric_counts = df.groupby('fabric_code').agg({
        'has_defect': ['sum', 'count']
    })
    fabric_counts.columns = ['Defect', 'No Defect']
    fabric_counts['No Defect'] = fabric_counts['No Defect'] - fabric_counts['Defect']

    fig, ax = plt.subplots(figsize=figsize)

    fabric_counts.plot(kind='bar', stacked=True, ax=ax,
                      color=['#e74c3c', '#2ecc71'])

    ax.set_xlabel('Fabric Code', fontsize=12, fontweight='bold')
    ax.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax.set_title('Distribution by Fabric Type', fontsize=14, fontweight='bold')
    ax.legend(title='Image Type', fontsize=10)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)

    plt.tight_layout()
    return fig


def plot_defect_size_distribution(df: pd.DataFrame, figsize: tuple[int, int] = (14, 5)):
    """
    Plot defect size distribution (area, width, height).

    Args:
        df: Dataset DataFrame with mask statistics
        figsize: Figure size
    """
    defect_df = df[df['has_defect']]

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Defect area
    axes[0].hist(defect_df['defect_area'].dropna(), bins=30,
                color='steelblue', edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('Defect Area (pixels)', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('Frequency', fontsize=11, fontweight='bold')
    axes[0].set_title('Defect Area Distribution', fontsize=12, fontweight='bold')
    axes[0].grid(alpha=0.3)

    # Width
    axes[1].hist(defect_df['bbox_width'].dropna(), bins=30,
                color='coral', edgecolor='black', alpha=0.7)
    axes[1].set_xlabel('Defect Width (pixels)', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('Frequency', fontsize=11, fontweight='bold')
    axes[1].set_title('Defect Width Distribution', fontsize=12, fontweight='bold')
    axes[1].grid(alpha=0.3)

    # Height
    axes[2].hist(defect_df['bbox_height'].dropna(), bins=30,
                color='mediumseagreen', edgecolor='black', alpha=0.7)
    axes[2].set_xlabel('Defect Height (pixels)', fontsize=11, fontweight='bold')
    axes[2].set_ylabel('Frequency', fontsize=11, fontweight='bold')
    axes[2].set_title('Defect Height Distribution', fontsize=12, fontweight='bold')
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    return fig


def plot_defect_size_by_type(df: pd.DataFrame, figsize: tuple[int, int] = (14, 6)):
    """
    Plot defect size comparison across defect types using boxplots.

    Args:
        df: Dataset DataFrame
        figsize: Figure size
    """
    defect_df = df[df['has_defect']]

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Area by defect type
    defect_df_sorted = defect_df.sort_values('defect_area', ascending=False)
    sns.boxplot(data=defect_df_sorted, y='defect_name', x='defect_area',
               ax=axes[0], palette='Set2', hue='defect_name', legend=False)
    axes[0].set_xlabel('Defect Area (pixels)', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('Defect Type', fontsize=11, fontweight='bold')
    axes[0].set_title('Defect Area by Type', fontsize=12, fontweight='bold')

    # Aspect ratio by defect type
    sns.boxplot(data=defect_df_sorted, y='defect_name', x='aspect_ratio',
               ax=axes[1], palette='Set3', hue='defect_name', legend=False)
    axes[1].set_xlabel('Aspect Ratio (width/height)', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('Defect Type', fontsize=11, fontweight='bold')
    axes[1].set_title('Defect Aspect Ratio by Type', fontsize=12, fontweight='bold')

    plt.tight_layout()
    return fig


def plot_fabric_defect_heatmap(matrix: pd.DataFrame, figsize: tuple[int, int] = (12, 8)):
    """
    Plot heatmap of fabric vs defect type correlation.

    Args:
        matrix: Cross-tabulation DataFrame
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(matrix, annot=True, fmt='d', cmap='YlOrRd',
               cbar_kws={'label': 'Count'}, ax=ax, linewidths=0.5)

    ax.set_xlabel('Defect Type', fontsize=12, fontweight='bold')
    ax.set_ylabel('Fabric Code', fontsize=12, fontweight='bold')
    ax.set_title('Fabric Type vs Defect Type Heatmap', fontsize=14, fontweight='bold')

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    return fig


def plot_spatial_heatmap(df: pd.DataFrame, figsize: tuple[int, int] = (14, 4)):
    """
    Plot spatial distribution heatmap of defects.

    Args:
        df: Dataset DataFrame with center coordinates
        figsize: Figure size
    """
    defect_df = df[df['has_defect']]

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # 2D histogram
    h, xedges, yedges = np.histogram2d(
        defect_df['center_x_norm'].dropna(),
        defect_df['center_y_norm'].dropna(),
        bins=20
    )

    im = axes[0].imshow(h.T, origin='lower', cmap='hot', aspect='auto',
                       extent=[0, 1, 0, 1], interpolation='bilinear')
    axes[0].set_xlabel('Normalized X Position', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('Normalized Y Position', fontsize=11, fontweight='bold')
    axes[0].set_title('Defect Spatial Distribution Heatmap', fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=axes[0], label='Frequency')

    # Scatter plot
    axes[1].scatter(defect_df['center_x_norm'], defect_df['center_y_norm'],
                   alpha=0.5, s=50, c=defect_df['defect_code'],
                   cmap='tab10', edgecolors='black', linewidths=0.5)
    axes[1].set_xlabel('Normalized X Position', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('Normalized Y Position', fontsize=11, fontweight='bold')
    axes[1].set_title('Defect Center Positions', fontsize=12, fontweight='bold')
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 1)
    axes[1].grid(alpha=0.3)

    # Horizontal and vertical marginal distributions
    axes[2].hist(defect_df['center_x_norm'].dropna(), bins=20, alpha=0.5,
                label='Horizontal', orientation='vertical', color='blue')
    axes[2].set_xlabel('Normalized Position', fontsize=11, fontweight='bold')
    axes[2].set_ylabel('Frequency', fontsize=11, fontweight='bold')
    axes[2].set_title('Position Distributions', fontsize=12, fontweight='bold')
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    return fig


def plot_interactive_defect_explorer(df: pd.DataFrame):
    """
    Create interactive Plotly visualization for defect exploration.

    Args:
        df: Dataset DataFrame
    """
    defect_df = df[df['has_defect']].copy()

    fig = px.scatter(
        defect_df,
        x='bbox_width',
        y='bbox_height',
        color='defect_name',
        size='defect_area',
        hover_data=['image_id', 'fabric_code', 'defect_percentage'],
        title='Interactive Defect Explorer: Size Analysis',
        labels={
            'bbox_width': 'Defect Width (pixels)',
            'bbox_height': 'Defect Height (pixels)',
            'defect_name': 'Defect Type'
        },
        template='plotly_white',
        width=1000,
        height=600
    )

    fig.update_traces(marker=dict(line=dict(width=1, color='DarkSlateGrey')))
    fig.update_layout(
        font=dict(size=12),
        title_font=dict(size=16, family='Arial Black')
    )

    return fig


def plot_defect_percentage_distribution(df: pd.DataFrame, figsize: tuple[int, int] = (12, 5)):
    """
    Plot defect coverage percentage distribution.

    Args:
        df: Dataset DataFrame
        figsize: Figure size
    """
    defect_df = df[df['has_defect']]

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Histogram
    axes[0].hist(defect_df['defect_percentage'].dropna(), bins=30,
                color='purple', edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('Defect Coverage (%)', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('Frequency', fontsize=11, fontweight='bold')
    axes[0].set_title('Defect Coverage Distribution', fontsize=12, fontweight='bold')
    axes[0].grid(alpha=0.3)

    # Violin plot by defect type
    top_defects = defect_df['defect_name'].value_counts().head(8).index
    filtered_df = defect_df[defect_df['defect_name'].isin(top_defects)]

    sns.violinplot(data=filtered_df, y='defect_name', x='defect_percentage',
                  ax=axes[1], palette='muted', hue='defect_name', legend=False)
    axes[1].set_xlabel('Defect Coverage (%)', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('Defect Type', fontsize=11, fontweight='bold')
    axes[1].set_title('Defect Coverage by Type (Top 8)', fontsize=12, fontweight='bold')

    plt.tight_layout()
    return fig


def plot_orientation_analysis(df: pd.DataFrame, figsize: tuple[int, int] = (12, 5)):
    """
    Plot defect orientation analysis.

    Args:
        df: Dataset DataFrame
        figsize: Figure size
    """
    defect_df = df[df['has_defect']].copy()

    def get_orientation(aspect_ratio):
        if pd.isna(aspect_ratio) or aspect_ratio == 0:
            return 'Unknown'
        elif aspect_ratio > 2:
            return 'Horizontal'
        elif aspect_ratio < 0.5:
            return 'Vertical'
        else:
            return 'Square'

    defect_df['orientation'] = defect_df['aspect_ratio'].apply(get_orientation)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Pie chart
    orientation_counts = defect_df['orientation'].value_counts()
    colors_pie = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
    axes[0].pie(orientation_counts.values, labels=orientation_counts.index,
               autopct='%1.1f%%', startangle=90, colors=colors_pie,
               textprops={'fontsize': 11, 'fontweight': 'bold'})
    axes[0].set_title('Defect Orientation Distribution', fontsize=12, fontweight='bold')

    # Aspect ratio distribution
    axes[1].hist(defect_df['aspect_ratio'].dropna(), bins=30,
                color='teal', edgecolor='black', alpha=0.7)
    axes[1].axvline(x=0.5, color='r', linestyle='--', label='Vertical threshold')
    axes[1].axvline(x=2.0, color='b', linestyle='--', label='Horizontal threshold')
    axes[1].set_xlabel('Aspect Ratio', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('Frequency', fontsize=11, fontweight='bold')
    axes[1].set_title('Aspect Ratio Distribution', fontsize=12, fontweight='bold')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    return fig
