"""
Evaluation metrics for Business Entity Resolution Challenge.
Implements the exact macro-averaged F_0.5 score per competition guidelines.
"""

from typing import Dict, Set, Optional, Tuple


def compute_entity_metrics(y_true: Set[str], y_pred: Set[str]) -> Tuple[float, float, float]:
    """
    Computes (Precision, Recall, F_0.5) for a single Source 1 entity.
    
    Rules per competition:
    - If true matches is empty (singleton) and prediction is empty: F_0.5 = 1.0
    - If true matches is empty and prediction is non-empty (false merge): F_0.5 = 0.0
    - If true matches is non-empty and prediction is empty: F_0.5 = 0.0
    - Otherwise: F_0.5 = (1.25 * P * R) / (0.25 * P + R)
    """
    if len(y_true) == 0:
        if len(y_pred) == 0:
            return 1.0, 1.0, 1.0
        else:
            return 0.0, 0.0, 0.0
    
    if len(y_pred) == 0:
        return 0.0, 0.0, 0.0
    
    tp = len(y_true & y_pred)
    precision = tp / len(y_pred)
    recall = tp / len(y_true)
    
    denom = 0.25 * precision + recall
    if denom == 0:
        f05 = 0.0
    else:
        f05 = (1.25 * precision * recall) / denom
        
    return precision, recall, f05


def evaluate_resolution(
    ground_truth: Dict[str, Set[str]],
    predictions: Dict[str, Set[str]],
    candidates: Optional[Dict[str, Set[str]]] = None
) -> Dict[str, float]:
    """
    Evaluates entity resolution predictions against ground truth across all S1 entities.
    Returns macro-averaged metrics and blocking diagnostics.
    """
    total_entities = len(ground_truth)
    if total_entities == 0:
        return {}

    f05_scores = []
    precision_scores = []
    recall_scores = []
    
    singleton_total = 0
    singleton_correct = 0
    
    non_singleton_f05 = []
    
    # Candidate/blocking diagnostics
    candidate_recalls = []
    candidate_counts = []

    for s1_id, y_true in ground_truth.items():
        y_pred = predictions.get(s1_id, set())
        
        p, r, f = compute_entity_metrics(y_true, y_pred)
        precision_scores.append(p)
        recall_scores.append(r)
        f05_scores.append(f)
        
        if len(y_true) == 0:
            singleton_total += 1
            if len(y_pred) == 0:
                singleton_correct += 1
        else:
            non_singleton_f05.append(f)
            
        if candidates is not None:
            cands = candidates.get(s1_id, set())
            candidate_counts.append(len(cands))
            if len(y_true) > 0:
                # Recall ceiling: what % of true matches were in candidate set
                cand_tp = len(y_true & cands)
                candidate_recalls.append(cand_tp / len(y_true))

    macro_f05 = sum(f05_scores) / total_entities
    macro_precision = sum(precision_scores) / total_entities
    macro_recall = sum(recall_scores) / total_entities
    
    results = {
        "macro_f05": macro_f05,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "total_entities": total_entities,
        "singleton_count": singleton_total,
        "singleton_accuracy": (singleton_correct / singleton_total) if singleton_total > 0 else 1.0,
        "non_singleton_macro_f05": (sum(non_singleton_f05) / len(non_singleton_f05)) if non_singleton_f05 else 0.0,
    }
    
    if candidates is not None:
        results["candidate_recall_ceiling"] = (sum(candidate_recalls) / len(candidate_recalls)) if candidate_recalls else 0.0
        results["avg_candidates_per_entity"] = (sum(candidate_counts) / len(candidate_counts)) if candidate_counts else 0.0

    return results
