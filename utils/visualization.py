import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_attention_heatmap(attention_matrix, tokens, title="Attention Heatmap", figsize=(10, 8)):
    """
    Визуализировать attention матрицу как тепловую карту
    Args:
        attention_matrix: numpy array [seq_len, seq_len]
        tokens: list токенов
        title: заголовок графика
        figsize: размер фигуры
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.heatmap(
        attention_matrix,
        xticklabels=tokens,
        yticklabels=tokens,
        cmap='viridis',
        ax=ax,
        cbar_kws={'label': 'Attention Weight'}
    )
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Key/Value Tokens', fontsize=12)
    ax.set_ylabel('Query Tokens', fontsize=12)
    
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    return fig

def compare_attention_patterns(bert_attention, gpt_attention, tokens, figsize=(16, 6)):
    """
    Сравнить attention patterns между двумя моделями
    Args:
        bert_attention: numpy array [seq_len, seq_len]
        gpt_attention: numpy array [seq_len, seq_len]
        tokens: list токенов
        figsize: размер фигуры
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # BERT (bidirectional)
    sns.heatmap(
        bert_attention,
        xticklabels=tokens,
        yticklabels=tokens,
        cmap='Blues',
        ax=axes[0],
        cbar_kws={'label': 'Attention Weight'}
    )
    axes[0].set_title('BERT (Bidirectional)', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Key/Value Tokens', fontsize=10)
    axes[0].set_ylabel('Query Tokens', fontsize=10)
    
    # GPT-2 (causal)
    sns.heatmap(
        gpt_attention,
        xticklabels=tokens,
        yticklabels=tokens,
        cmap='Oranges',
        ax=axes[1],
        cbar_kws={'label': 'Attention Weight'}
    )
    axes[1].set_title('GPT-2 (Causal)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Key/Value Tokens', fontsize=10)
    axes[1].set_ylabel('Query Tokens', fontsize=10)
    
    for ax in axes:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    return fig

def plot_all_heads(attention_heads, tokens, layer_idx=0, figsize=(16, 12)):
    """
    Визуализировать все attention головы для слоя
    Args:
        attention_heads: numpy array [num_heads, seq_len, seq_len]
        tokens: list токенов
        layer_idx: индекс слоя (для заголовка)
        figsize: размер фигуры
    """
    num_heads = attention_heads.shape[0]
    cols = 4
    rows = (num_heads + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten()
    
    for head_idx in range(num_heads):
        sns.heatmap(
            attention_heads[head_idx],
            xticklabels=tokens if head_idx % cols == 0 else False,
            yticklabels=tokens if head_idx % cols == 0 else False,
            cmap='YlOrRd',
            ax=axes[head_idx],
            cbar=False
        )
        axes[head_idx].set_title(f'Head {head_idx}', fontsize=10)
        
        if head_idx % cols != 0:
            axes[head_idx].set_ylabel('')
    
    for idx in range(num_heads, len(axes)):
        axes[idx].remove()
    
    plt.suptitle(f'All Attention Heads - Layer {layer_idx}', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    return fig
