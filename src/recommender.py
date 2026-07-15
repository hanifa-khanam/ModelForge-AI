"""
Applies weighted scoring to model evaluation results and produces a final recommendation with human-readable reasoning.
"""

from src.config import PRIMARY_METRIC, RECOMMENDATION_WEIGHTS, OVERFITTING_THRESHOLD, OVERFITTING_PENALTY

def recommend_model(evaluation: dict, training_results: dict, prefer_fast: bool=False) -> dict:
    """
    Recommend the best model based on composite scoring
    Args: 
        evaluation: output from evaluator.evaluate_models() 
        training_results: output from trainer.train_models() 
        prefer_fast: If true, weight speed equally with performance
    returns:
        dict with recommended_model, composite_score, reasoning, all_scores
    """
    
    problem_type = evaluation["problem_type"]
    primary_metric = PRIMARY_METRIC[problem_type]
    all_results = training_results["results"]
    
    # Select weights
    weight_key = "balanced" if prefer_fast else "accuracy_focused"
    weights = RECOMMENDATION_WEIGHTS[weight_key]
    
    # Normalization bounds
    best_metric = max(r["metrics"][primary_metric] for r in all_results)
    fastest_time = min(r["training_time"] for r in all_results)
    slowest_time = max(r["training_time"] for r in all_results)
    
    scored_models = []
    
    for result in all_results:
        model_name = result["model_name"]
        test_metric = result["metrics"][primary_metric]
        train_metric = result.get("train_metrics", {}).get(primary_metric, test_metric)
        training_time = result.get("training_time")
        
        # Performance score (0-100)
        if best_metric > 0:
            performance_score = (test_metric / best_metric) * 100
        else:
            performance_score = 0
        
        # Speed score (0-100, faster = higher)
        if training_time and slowest_time > 0:
            speed_score = (fastest_time / training_time) * 100
        else:
            speed_score = 50
        
        # Overfitting penalty
        gap = train_metric - test_metric
        overfitting_penalty = (
            OVERFITTING_PENALTY if gap > OVERFITTING_THRESHOLD else 0
        )
        
        # Composite
        composite = (
            (weights["performance"] * performance_score) +
            (weights["speed"] * speed_score) +
            overfitting_penalty
        )
        
        scored_models.append({
            "model_name": model_name,
            "composite": round(composite, 2),
            "performance_score": round(performance_score, 2),
            "speed_score": round(speed_score, 2),
            "overfitting_penalty": overfitting_penalty,
            "is_overfitting": gap > OVERFITTING_THRESHOLD,
            "primary_metric_value": test_metric,
            "training_time": training_time,
        })
    
    # Sort by composite (descending)
    scored_models.sort(key=lambda m: m["composite"], reverse=True)
    best = scored_models[0]
    
    # Build reasoning
    reasoning = [
        f"{best['model_name']} achieves the highest composite score "
        f"({best['composite']})."
    ]
    
    if best["performance_score"] < 100:
        reasoning.append(
            f"{primary_metric.upper()} of {best['primary_metric_value']:.3f} "
            f"is {100 - best['performance_score']:.1f}% below the best performer."
        )
    else:
        reasoning.append(
            f"Best {primary_metric.upper()} score: {best['primary_metric_value']:.3f}."
        )
    
    if best["training_time"]:
        reasoning.append(
            f"Training time of {best['training_time']:.2f}s "
            f"(speed score: {best['speed_score']:.0f}/100)."
        )
    
    if best["is_overfitting"]:
        reasoning.append(
            "Warning: Train-test gap detected. Model may be overfitting."
        )
    else:
        reasoning.append("No significant overfitting detected.")
    
    reasoning.append(
        f"Weighting: {weights['performance']*100:.0f}% performance, "
        f"{weights['speed']*100:.0f}% speed."
    )
    
    return {
        "recommended_model": best["model_name"],
        "composite_score": best["composite"],
        "reasoning": reasoning,
        "all_scores": scored_models,
        "weight_mode": weight_key,
    }
    
