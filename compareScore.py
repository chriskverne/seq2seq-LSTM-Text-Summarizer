from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import numpy as np
from tqdm import tqdm
import re

def load_summaries(filepath):
    """
    Load generated and reference summaries from the simplified output file.
    Returns two lists: generated summaries and reference summaries.
    """
    generated_summaries = []
    reference_summaries = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        current_type = None
        current_text = ''
        
        for line in f:
            line = line.strip()
            
            if line.startswith('REAL SUMMARY:'):
                if current_type == 'generated' and current_text:
                    generated_summaries.append(current_text.strip())
                current_type = 'real'
                current_text = line[len('REAL SUMMARY:'):].strip()
                if current_text:
                    reference_summaries.append(current_text)
                    current_text = ''
                
            elif line.startswith('GENERATED SUMMARY:'):
                if current_type == 'real' and current_text:
                    reference_summaries.append(current_text.strip())
                current_type = 'generated'
                current_text = line[len('GENERATED SUMMARY:'):].strip()
                if current_text:
                    generated_summaries.append(current_text)
                    current_text = ''
                
            elif line:  # Continue current summary
                if current_type == 'real':
                    if not current_text:  # If this is the first line after the header
                        reference_summaries.append(line)
                    else:
                        current_text += ' ' + line
                elif current_type == 'generated':
                    if not current_text:  # If this is the first line after the header
                        generated_summaries.append(line)
                    else:
                        current_text += ' ' + line
    
    # Add the last summary if there is one
    if current_type == 'real' and current_text:
        reference_summaries.append(current_text.strip())
    elif current_type == 'generated' and current_text:
        generated_summaries.append(current_text.strip())
    
    return generated_summaries, reference_summaries

def analyze_generation_patterns(generated_summaries):
    """
    Analyze common patterns and issues in the generated summaries.
    """
    patterns = {
        'repetition': 0,  # Count summaries with obvious word repetitions
        'fire_mentions': 0,  # Count summaries mentioning 'fire'
        'crash_mentions': 0,  # Count summaries mentioning 'crash'
        'london_mentions': 0,  # Count summaries with london/londonderry
        'number_phrases': 0,  # Count phrases like "at least people"
    }
    
    for summary in generated_summaries:
        # Check for word repetitions
        words = summary.split()
        for i in range(len(words)-2):
            if words[i] == words[i+1] == words[i+2]:
                patterns['repetition'] += 1
                break
        
        # Check for specific patterns
        if 'fire' in summary.lower():
            patterns['fire_mentions'] += 1
        if 'crash' in summary.lower():
            patterns['crash_mentions'] += 1
        if 'london' in summary.lower():
            patterns['london_mentions'] += 1
        if 'at least people' in summary.lower():
            patterns['number_phrases'] += 1
    
    return patterns

def calculate_bleu(generated_summaries, reference_summaries):
    """
    Calculate corpus BLEU score for the generated summaries.
    """
    # Tokenize summaries
    generated_tokens = [summary.split() for summary in generated_summaries]
    reference_tokens = [[summary.split()] for summary in reference_summaries]
    
    # Calculate BLEU scores for different n-grams
    smoothing = SmoothingFunction().method1
    bleu_scores = {
        'bleu-1': corpus_bleu(reference_tokens, generated_tokens, weights=(1, 0, 0, 0), smoothing_function=smoothing),
        'bleu-2': corpus_bleu(reference_tokens, generated_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing),
        'bleu-3': corpus_bleu(reference_tokens, generated_tokens, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothing),
        'bleu-4': corpus_bleu(reference_tokens, generated_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing)
    }
    
    return bleu_scores

def calculate_rouge(generated_summaries, reference_summaries):
    """
    Calculate ROUGE scores for the generated summaries.
    """
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    scores = {
        'rouge1': {'precision': [], 'recall': [], 'fmeasure': []},
        'rouge2': {'precision': [], 'recall': [], 'fmeasure': []},
        'rougeL': {'precision': [], 'recall': [], 'fmeasure': []}
    }
    
    for gen, ref in tqdm(zip(generated_summaries, reference_summaries), desc="Calculating ROUGE scores"):
        score = scorer.score(ref, gen)
        
        for metric in ['rouge1', 'rouge2', 'rougeL']:
            scores[metric]['precision'].append(score[metric].precision)
            scores[metric]['recall'].append(score[metric].recall)
            scores[metric]['fmeasure'].append(score[metric].fmeasure)
    
    # Calculate averages
    avg_scores = {}
    for metric in scores:
        avg_scores[metric] = {
            'precision': np.mean(scores[metric]['precision']),
            'recall': np.mean(scores[metric]['recall']),
            'fmeasure': np.mean(scores[metric]['fmeasure'])
        }
    
    return avg_scores

def analyze_summaries(generated_summaries):
    """
    Analyze basic statistics of the generated summaries.
    """
    lengths = [len(summary.split()) for summary in generated_summaries]
    stats = {
        'count': len(generated_summaries),
        'avg_length': np.mean(lengths),
        'std_length': np.std(lengths),
        'min_length': np.min(lengths),
        'max_length': np.max(lengths),
        'empty_summaries': sum(1 for s in generated_summaries if not s.strip()),
        'unique_summaries': len(set(generated_summaries))
    }
    return stats

def main():
    filepath = './summaries2.txt'
    
    print("Loading summaries...")
    generated_summaries, reference_summaries = load_summaries(filepath)
    
    print(f"\nLoaded {len(generated_summaries)} generated summaries and {len(reference_summaries)} reference summaries")
    
    print("\nAnalyzing summary statistics...")
    stats = analyze_summaries(generated_summaries)
    print("\nSummary Statistics:")
    print(f"Total summaries: {stats['count']}")
    print(f"Average length: {stats['avg_length']:.2f} words")
    print(f"Length std dev: {stats['std_length']:.2f} words")
    print(f"Min length: {stats['min_length']} words")
    print(f"Max length: {stats['max_length']} words")
    print(f"Empty summaries: {stats['empty_summaries']}")
    print(f"Unique summaries: {stats['unique_summaries']}")
    
    print("\nAnalyzing generation patterns...")
    patterns = analyze_generation_patterns(generated_summaries)
    print("\nGeneration Patterns:")
    print(f"Summaries with word repetitions: {patterns['repetition']}")
    print(f"Summaries mentioning 'fire': {patterns['fire_mentions']}")
    print(f"Summaries mentioning 'crash': {patterns['crash_mentions']}")
    print(f"Summaries mentioning 'london': {patterns['london_mentions']}")
    print(f"Summaries with 'at least people' phrase: {patterns['number_phrases']}")
    
    print("\nCalculating BLEU scores...")
    bleu_scores = calculate_bleu(generated_summaries, reference_summaries)
    print("\nBLEU Scores:")
    for metric, score in bleu_scores.items():
        print(f"{metric}: {score:.4f}")
    
    print("\nCalculating ROUGE scores...")
    rouge_scores = calculate_rouge(generated_summaries, reference_summaries)
    
    print("\nROUGE Scores:")
    for metric, scores in rouge_scores.items():
        print(f"\n{metric}:")
        print(f"Precision: {scores['precision']:.4f}")
        print(f"Recall: {scores['recall']:.4f}")
        print(f"F1: {scores['fmeasure']:.4f}")

if __name__ == "__main__":
    main()