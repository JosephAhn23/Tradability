"""
Answer The Question: Is this framework a tool for finding edge, or rationalization for inaction?

Uses empirical metrics (correlation, AUC) to definitively answer.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
from prove_selectivity import create_signal_spectrum, SignalConfig


def assign_ground_truth(signals: List[SignalConfig]) -> Dict[str, int]:
    """
    Assign ground truth: 1 = actually good signal, 0 = bad signal.
    
    Ground truth is based on:
    - Gross Sharpe >= 0.6 (before costs, signal has real edge)
    - Turnover <= 2.0 (practical to trade)
    
    This is a more lenient bar than what the framework uses,
    representing "signals a reasonable quant would consider good."
    """
    ground_truth = {}
    
    for signal in signals:
        # "Truly good" = decent gross sharpe AND not insane turnover
        is_good = signal.gross_sharpe >= 0.6 and signal.turnover <= 2.0
        ground_truth[signal.name] = 1 if is_good else 0
    
    return ground_truth


def get_framework_verdicts(signals: List[SignalConfig], cost_bps: float = 10.0) -> Dict[str, int]:
    """Get framework's PASS/REJECT decisions as binary."""
    verdicts = {}
    vol = 0.15
    
    for signal in signals:
        annual_cost = signal.turnover * (cost_bps / 10000) * 2
        net_sharpe = signal.gross_sharpe - (annual_cost / vol)
        
        passes_sharpe = net_sharpe >= 0.5
        passes_turnover = signal.turnover <= 3.0
        decision = 1 if (passes_sharpe and passes_turnover) else 0
        
        verdicts[signal.name] = decision
    
    return verdicts


def calculate_discrimination_metrics(y_true: List[int], y_pred: List[int]) -> Dict:
    """Calculate how well framework discriminates good from bad."""
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Key metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # AUC (requires probabilities, so we use binary predictions as proxy)
    try:
        auc = roc_auc_score(y_true, y_pred)
    except ValueError:
        auc = 0.5  # If only one class present
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'auc': auc,
        'true_positives': tp,
        'false_positives': fp,
        'true_negatives': tn,
        'false_negatives': fn
    }


def calculate_net_sharpe_scores(signals: List[SignalConfig], cost_bps: float) -> Dict[str, float]:
    """Get net Sharpe as continuous score for better AUC calculation."""
    scores = {}
    vol = 0.15
    
    for signal in signals:
        annual_cost = signal.turnover * (cost_bps / 10000) * 2
        net_sharpe = signal.gross_sharpe - (annual_cost / vol)
        scores[signal.name] = net_sharpe
    
    return scores


def run_empirical_test():
    """Main test: Does framework discriminate good from bad?"""
    print("=" * 70)
    print("EMPIRICAL TEST: Does the framework discriminate signal quality?")
    print("=" * 70)
    
    signals = create_signal_spectrum()
    ground_truth = assign_ground_truth(signals)
    
    # Test at three cost levels
    cost_levels = {
        'realistic': 1.5,
        'conservative': 10.0,
        'sabotage': 50.0
    }
    
    results = []
    
    for level_name, cost_bps in cost_levels.items():
        print(f"\n### {level_name.upper()} Costs ({cost_bps} bps) ###\n")
        
        verdicts = get_framework_verdicts(signals, cost_bps)
        net_sharpes = calculate_net_sharpe_scores(signals, cost_bps)
        
        # Align arrays
        names = list(ground_truth.keys())
        y_true = [ground_truth[n] for n in names]
        y_pred = [verdicts[n] for n in names]
        y_scores = [net_sharpes[n] for n in names]
        
        # Binary metrics
        metrics = calculate_discrimination_metrics(y_true, y_pred)
        
        # Continuous AUC (using net Sharpe as score)
        try:
            continuous_auc = roc_auc_score(y_true, y_scores)
        except ValueError:
            continuous_auc = 0.5
        
        # Correlation between ground truth and net Sharpe
        correlation = np.corrcoef(y_true, y_scores)[0, 1]
        
        print(f"Confusion Matrix:")
        print(f"  True Positives (good signals passed):   {metrics['true_positives']}")
        print(f"  False Positives (bad signals passed):   {metrics['false_positives']}")
        print(f"  True Negatives (bad signals rejected):  {metrics['true_negatives']}")
        print(f"  False Negatives (good signals rejected):{metrics['false_negatives']}")
        print()
        print(f"Metrics:")
        print(f"  Accuracy:     {metrics['accuracy']:.2%}")
        print(f"  Precision:    {metrics['precision']:.2%}")
        print(f"  Recall:       {metrics['recall']:.2%}")
        print(f"  Specificity:  {metrics['specificity']:.2%}")
        print(f"  Binary AUC:   {metrics['auc']:.3f}")
        print(f"  Continuous AUC (net Sharpe): {continuous_auc:.3f}")
        print(f"  Correlation:  {correlation:.3f}")
        
        results.append({
            'cost_level': level_name,
            'cost_bps': cost_bps,
            **metrics,
            'continuous_auc': continuous_auc,
            'correlation': correlation
        })
    
    results_df = pd.DataFrame(results)
    
    # FINAL VERDICT
    print("\n" + "=" * 70)
    print("FINAL VERDICT")
    print("=" * 70)
    
    # Use conservative level for judgment
    conservative_result = results_df[results_df['cost_level'] == 'conservative'].iloc[0]
    
    auc_score = conservative_result['continuous_auc']
    correlation = conservative_result['correlation']
    recall = conservative_result['recall']
    specificity = conservative_result['specificity']
    
    print(f"\nAt conservative costs (10 bps):")
    print(f"  AUC = {auc_score:.3f}")
    print(f"  Correlation = {correlation:.3f}")
    print(f"  Recall (catches good signals) = {recall:.2%}")
    print(f"  Specificity (rejects bad signals) = {specificity:.2%}")
    print()
    
    # Scoring logic
    is_tool = False
    reasons = []
    
    if auc_score >= 0.7:
        reasons.append(f"AUC >= 0.7 ({auc_score:.3f}): Discriminates well")
        is_tool = True
    elif auc_score >= 0.6:
        reasons.append(f"AUC 0.6-0.7 ({auc_score:.3f}): Moderate discrimination")
    else:
        reasons.append(f"AUC < 0.6 ({auc_score:.3f}): Poor discrimination")
    
    if correlation >= 0.5:
        reasons.append(f"Correlation >= 0.5 ({correlation:.3f}): Net Sharpe aligns with quality")
    else:
        reasons.append(f"Correlation < 0.5 ({correlation:.3f}): Weak alignment")
    
    if recall >= 0.5:
        reasons.append(f"Recall >= 50% ({recall:.1%}): Catches most good signals")
    else:
        reasons.append(f"Recall < 50% ({recall:.1%}): Misses too many good signals")
        is_tool = False
    
    if specificity >= 0.8:
        reasons.append(f"Specificity >= 80% ({specificity:.1%}): Rejects most bad signals")
    else:
        reasons.append(f"Specificity < 80% ({specificity:.1%}): Lets through bad signals")
    
    print("Assessment:")
    for reason in reasons:
        print(f"  - {reason}")
    print()
    
    if is_tool and recall >= 0.3 and specificity >= 0.7:
        verdict = "TOOL_FOR_FINDING_EDGE"
        print(f"VERDICT: {verdict}")
        print("The framework discriminates between good and bad signals.")
        print("It is a USEFUL TOOL for evaluating trading strategies.")
    elif specificity >= 0.9 and recall < 0.3:
        verdict = "OVER_CONSERVATIVE_BUT_SAFE"
        print(f"VERDICT: {verdict}")
        print("The framework is too conservative - rejects too many good signals.")
        print("Consider lowering thresholds, but it's not 'rationalizing inaction'.")
    else:
        verdict = "NEEDS_RECALIBRATION"
        print(f"VERDICT: {verdict}")
        print("The framework needs recalibration to better balance precision and recall.")
    
    print("\n" + "=" * 70)
    
    # Save
    results_df.to_csv('empirical_test_results.csv', index=False)
    print(f"Results saved to empirical_test_results.csv")
    
    # Create detailed signal-level report
    signal_df = pd.DataFrame([
        {
            'name': s.name,
            'category': s.category,
            'gross_sharpe': s.gross_sharpe,
            'turnover': s.turnover,
            'ground_truth': ground_truth[s.name],
            'framework_verdict': get_framework_verdicts(signals, 10.0)[s.name],
            'net_sharpe': calculate_net_sharpe_scores(signals, 10.0)[s.name]
        }
        for s in signals
    ])
    signal_df['correct'] = signal_df['ground_truth'] == signal_df['framework_verdict']
    signal_df.to_csv('signal_level_assessment.csv', index=False)
    print(f"Signal-level details saved to signal_level_assessment.csv")
    
    return verdict, results_df


if __name__ == "__main__":
    run_empirical_test()
