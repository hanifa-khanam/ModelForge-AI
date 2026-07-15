"""
Evaluator Module - Model comparison and Leaderboard Generation
"""

from src.config import PRIMARY_METRIC, SECONDARY_METRIC

def evaluate_models(training_results: dict) -> dict:
    """
    Compare trained models and produce a ranked leaderboard.
    
    Arg:
    taining_results: Output from trainer.train_models()
    Returns:
    dict with best_model, leaderboard, and summary
    """
    
    problem_type = training_results["problem_type"] 
    results = training_results["results"] 
    
    # get the right metric names for this problem type
    primary_metric = PRIMARY_METRIC[problem_type]
    secondary_metric = SECONDARY_METRIC[problem_type]
    
    # sort by primary metric (descending), tiebreak by secondary
    sorted_results = sorted(results, key=lambda r: (r["metrics"][primary_metric], r["metrics"][secondary_metric]), reverse=True)
    
    # build ranked leaderboard
    leaderboard = [] 
    for rank, result in enumerate(sorted_results, start=1):
        leaderboard.append({
            "rank" : rank,
            "model_name" : result["model_name"],
            "metrics" : result["metrics"],
            "training_time" : result["training_time"]
        })
        
        best_model = leaderboard[0]
    
    # Generate summary
    worst_model = leaderboard[-1] 
    best_score = best_model["metrics"][primary_metric] 
    worst_score = worst_model["metrics"][primary_metric] 
    
    
    summary = (
        f"{best_model['model_name']} leads with "
        f"{primary_metric.upper()}={best_score:.3f}. " 
        f"Worst performer is {worst_model['model_name']} "
        f"with {primary_metric.upper()}={worst_score:.3f}. " 
    )
    
    return {
        "problem_type": problem_type,
        "primary_metric": primary_metric,
        "best_model": {
            "name": best_model["model_name"],
            "metrics": best_model["metrics"],
            "training_time": best_model["training_time"]
        },
        "leaderboard": leaderboard,
        "summary": summary
    }
    
def compare_models(training_results: dict) -> dict:
    """
    Analyze relative performance between models.
    """
    
    problem_type = training_results["problem_type"] 
    results = training_results["results"] 
    primary_metric = PRIMARY_METRIC[problem_type] 
    
    if len(results) < 2:
        return {
            "best_worst_gap": 0,
            "is_dominant": False,
            "summary": "Only one model trained. No comparison possible."
        }
    
    # Find best and worst
    best = max(results, key=lambda r: r["metrics"][primary_metric])
    worst = min(results, key=lambda r: r["metrics"][primary_metric])
    
    best_score = best["metrics"][primary_metric]
    worst_score = worst["metrics"][primary_metric]
    gap = best_score - worst_score
    
    # Check if best model wins across ALL metrics
    metric_names = list(best["metrics"].keys())
    is_dominant = True
    
    for metric in metric_names:
        best_value = best["metrics"][metric]
        for result in results:
            if result["model_name"] != best["model_name"]:
                # For all metrics, higher is better
                if result["metrics"][metric] > best_value:
                    is_dominant = False
                    break
        if not is_dominant:
            break
    
    if is_dominant:
        comparison_summary = (
            f"{best['model_name']} dominates across all metrics. "
            f"It is the clear recommendation."
        )
    else:
        comparison_summary = (
            f"No single model dominates all metrics. "
            f"{best['model_name']} leads on {primary_metric.upper()} "
            f"by a margin of {gap:.3f}."
        )
    
    return {
        "best_model_name": best["model_name"],
        "worst_model_name": worst["model_name"],
        "best_worst_gap": round(gap, 4),
        "is_dominant": is_dominant,
        "summary": comparison_summary
    }