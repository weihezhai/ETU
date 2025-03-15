import json
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import seaborn as sns
import argparse
from pathlib import Path


def load_jsonl(file_path):
    """Load data from a JSONL file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def extract_features(data):
    """
    Extract features from the data:
    - similarity_score for each path
    - is_hit: whether the end node is in ground truth
    """
    all_features = []
    
    for item in data:
        ground_truth = set(item["ground_truth"])
        
        # Count number of paths per question
        num_paths = len(item["paths"])
        
        for path in item["paths"]:
            # Get the end node of the path
            end_node = path["path"][-1]
            
            # Check if the end node is in ground truth
            is_hit = 1 if end_node in ground_truth else 0
            
            # Extract features
            features = {
                "id": item["id"],
                "question": item["question"],
                "similarity_score": path["similarity_score"],
                "path_str": path["path_str"],
                "end_node": end_node,
                "is_hit": is_hit,
                "ppl_score": path.get("ppl_score", None),  # Include PPL score if available
                "question_id": item["id"],  # Add question_id for question-level analysis
                "num_paths": num_paths  # Add number of paths per question
            }
            
            all_features.append(features)
    
    # Create the DataFrame
    df = pd.DataFrame(all_features)
    
    # Add normalized similarity scores within each question
    df["normalized_similarity"] = df.groupby("question_id")["similarity_score"].transform(
        lambda x: (x - x.min()) / (x.max() - x.min()) if x.max() != x.min() else 0.5
    )
    
    # Add rank of similarity score within each question (higher score = higher rank)
    df["similarity_rank_in_question"] = df.groupby("question_id")["similarity_score"].transform(
        lambda x: x.rank(method="dense", ascending=False)
    )
    
    # Calculate if a path has the highest similarity score in its question
    df["is_highest_similarity"] = df.groupby("question_id")["similarity_score"].transform(
        lambda x: x == x.max()
    ).astype(int)
    
    # Calculate if a question has any hit paths
    df["question_has_hit"] = df.groupby("question_id")["is_hit"].transform("max")
    
    # Calculate total hits per question
    df["question_total_hits"] = df.groupby("question_id")["is_hit"].transform("sum")
    
    # Calculate hit ratio per question (hits / total paths)
    df["question_hit_ratio"] = df["question_total_hits"] / df["num_paths"]
    
    return df


def statistical_analysis(df):
    """Perform statistical analysis on the data."""
    results = {}
    
    # Get basic counts
    results["total_questions"] = df["question_id"].nunique()
    results["questions_with_one_path"] = df[df["num_paths"] == 1]["question_id"].nunique()
    results["questions_with_all_unhit"] = df[df["question_has_hit"] == 0]["question_id"].nunique()
    
    # Split data into hit and no-hit groups
    hit_scores = df[df["is_hit"] == 1]["similarity_score"]
    no_hit_scores = df[df["is_hit"] == 0]["similarity_score"]
    
    # Basic statistics
    results["hit_mean"] = hit_scores.mean() if not hit_scores.empty else float('nan')
    results["hit_std"] = hit_scores.std() if not hit_scores.empty else float('nan')
    results["no_hit_mean"] = no_hit_scores.mean() if not no_hit_scores.empty else float('nan')
    results["no_hit_std"] = no_hit_scores.std() if not no_hit_scores.empty else float('nan')
    
    # Statistical tests only if we have both hit and no-hit groups
    if not hit_scores.empty and not no_hit_scores.empty:
        # T-test to check if the difference is significant
        t_stat, p_value = stats.ttest_ind(hit_scores, no_hit_scores, equal_var=False)
        results["t_stat"] = t_stat
        results["p_value"] = p_value
        
        # Mann-Whitney U test (non-parametric)
        u_stat, u_p_value = stats.mannwhitneyu(hit_scores, no_hit_scores)
        results["u_stat"] = u_stat
        results["u_p_value"] = u_p_value
        
        # Point-biserial correlation
        corr, corr_p_value = stats.pointbiserialr(df["is_hit"], df["similarity_score"])
        results["correlation"] = corr
        results["corr_p_value"] = corr_p_value
        
        # Logistic regression
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import classification_report
    
        X = df[["similarity_score"]]
        y = df["is_hit"]
        
        logistic_model = LogisticRegression()
        logistic_model.fit(X, y)
        
        results["logistic_coef"] = logistic_model.coef_[0][0]
        results["logistic_intercept"] = logistic_model.intercept_[0]
        
        # Calculate odds ratio
        odds_ratio = np.exp(logistic_model.coef_[0][0])
        results["odds_ratio"] = odds_ratio
    else:
        # If we don't have both hit and no-hit groups, set results to NaN
        results["t_stat"] = float('nan')
        results["p_value"] = float('nan')
        results["u_stat"] = float('nan')
        results["u_p_value"] = float('nan')
        results["correlation"] = float('nan')
        results["corr_p_value"] = float('nan')
        results["logistic_coef"] = float('nan')
        results["logistic_intercept"] = float('nan')
        results["odds_ratio"] = float('nan')
    
    # ---- QUESTION-LEVEL ANALYSIS ----
    
    # Filter out single-path questions for certain analyses
    multi_path_df = df[df["num_paths"] > 1]
    
    # Calculate accuracy of using highest similarity to predict hit
    # For all questions
    highest_sim_paths = df[df["is_highest_similarity"] == 1]
    results["highest_sim_accuracy"] = highest_sim_paths["is_hit"].mean() if not highest_sim_paths.empty else float('nan')
    
    # For multi-path questions only
    highest_sim_multi = multi_path_df[multi_path_df["is_highest_similarity"] == 1]
    results["highest_sim_accuracy_multi"] = highest_sim_multi["is_hit"].mean() if not highest_sim_multi.empty else float('nan')
    
    # For questions with at least one hit path
    questions_with_hits = df[df["question_has_hit"] == 1]
    hit_questions_ids = questions_with_hits["question_id"].unique()
    results["questions_with_hits_count"] = len(hit_questions_ids)
    results["total_questions_count"] = df["question_id"].nunique()
    
    if len(hit_questions_ids) > 0:
        # Analysis on questions with hits
        hit_questions = df[df["question_id"].isin(hit_questions_ids)]
        hit_top_paths = hit_questions[hit_questions["is_highest_similarity"] == 1]
        
        # How often is the highest similarity path a hit for questions with hits?
        results["hit_questions_highest_sim_accuracy"] = hit_top_paths["is_hit"].mean()
        
        # Average rank of hit paths within their questions
        hit_paths = hit_questions[hit_questions["is_hit"] == 1]
        results["avg_hit_path_rank"] = hit_paths["similarity_rank_in_question"].mean()
        
        # For multi-path questions with hits only
        multi_path_hit_questions = hit_questions[hit_questions["num_paths"] > 1]
        if not multi_path_hit_questions.empty:
            multi_path_hit_top = multi_path_hit_questions[multi_path_hit_questions["is_highest_similarity"] == 1]
            results["hit_questions_highest_sim_accuracy_multi"] = multi_path_hit_top["is_hit"].mean()
    
    # Analysis on normalized scores
    if not df["is_hit"].empty and not df["normalized_similarity"].empty:
        results["norm_correlation"], results["norm_corr_p_value"] = stats.pointbiserialr(
            df["is_hit"], df["normalized_similarity"]
        )
    else:
        results["norm_correlation"] = float('nan')
        results["norm_corr_p_value"] = float('nan')
    
    return results


def plot_results(df, output_dir):
    """Generate plots to visualize the results."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Create a multi-path only DataFrame for some analyses
    multi_path_df = df[df["num_paths"] > 1]
    
    # Histogram of similarity scores by hit/no-hit
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x="similarity_score", hue="is_hit", 
                 element="step", stat="density", common_norm=False)
    plt.title("Distribution of Similarity Scores by Hit/No-Hit")
    plt.xlabel("Similarity Score")
    plt.ylabel("Density")
    plt.savefig(output_dir / "similarity_score_distribution.png", dpi=300, bbox_inches="tight")
    
    # Only create ROC curve if we have both hit and no-hit instances
    if df["is_hit"].nunique() > 1:
        # ROC curve
        fpr, tpr, thresholds = roc_curve(df["is_hit"], df["similarity_score"])
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(10, 6))
        plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (area = {roc_auc:.2f})")
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver Operating Characteristic (ROC)")
        plt.legend(loc="lower right")
        plt.savefig(output_dir / "roc_curve.png", dpi=300, bbox_inches="tight")
        
        # Precision-Recall curve
        precision, recall, _ = precision_recall_curve(df["is_hit"], df["similarity_score"])
        pr_auc = auc(recall, precision)
        
        plt.figure(figsize=(10, 6))
        plt.plot(recall, precision, color="blue", lw=2, label=f"PR curve (area = {pr_auc:.2f})")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")
        plt.legend(loc="lower left")
        plt.savefig(output_dir / "precision_recall_curve.png", dpi=300, bbox_inches="tight")
        
        # Boxplot
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df, x="is_hit", y="similarity_score")
        plt.title("Similarity Score by Hit/No-Hit")
        plt.xlabel("Is Hit (1=Yes, 0=No)")
        plt.ylabel("Similarity Score")
        plt.savefig(output_dir / "similarity_score_boxplot.png", dpi=300, bbox_inches="tight")
    
    # If PPL score is available, plot its relationship with similarity score
    if "ppl_score" in df.columns and not df["ppl_score"].isna().all():
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df, x="similarity_score", y="ppl_score", hue="is_hit")
        plt.title("Similarity Score vs PPL Score")
        plt.xlabel("Similarity Score")
        plt.ylabel("PPL Score")
        plt.savefig(output_dir / "similarity_vs_ppl.png", dpi=300, bbox_inches="tight")
    
    # --- QUESTION-LEVEL VISUALIZATIONS ---
    
    # Distribution of normalized similarity scores
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x="normalized_similarity", hue="is_hit", 
                 element="step", stat="density", common_norm=False)
    plt.title("Distribution of Normalized Similarity Scores by Hit/No-Hit")
    plt.xlabel("Normalized Similarity Score")
    plt.ylabel("Density")
    plt.savefig(output_dir / "normalized_similarity_distribution.png", dpi=300, bbox_inches="tight")
    
    # Path count distribution
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df.drop_duplicates("question_id"), x="num_paths")
    plt.title("Distribution of Path Count per Question")
    plt.xlabel("Number of Paths")
    plt.ylabel("Count of Questions")
    plt.savefig(output_dir / "path_count_distribution.png", dpi=300, bbox_inches="tight")
    
    # For multi-path questions only
    if not multi_path_df.empty:
        # Boxplot of rank within question for hit vs non-hit paths
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=multi_path_df, x="is_hit", y="similarity_rank_in_question")
        plt.title("Rank of Path Within Question by Hit/No-Hit\n(Multi-path Questions Only)")
        plt.xlabel("Is Hit (1=Yes, 0=No)")
        plt.ylabel("Rank Within Question (lower is better)")
        plt.savefig(output_dir / "rank_in_question_boxplot.png", dpi=300, bbox_inches="tight")
    
    # Heatmap of highest similarity path being a hit
    plt.figure(figsize=(10, 6))
    highest_sim_paths = df[df["is_highest_similarity"] == 1]
    
    # Check if we have both hit and no-hit questions and both hit and no-hit paths
    if highest_sim_paths["question_has_hit"].nunique() > 1 and highest_sim_paths["is_hit"].nunique() > 1:
        highest_sim_hit_rate = pd.crosstab(
            highest_sim_paths["is_hit"], 
            highest_sim_paths["question_has_hit"], 
            normalize="columns"
        )
        
        if highest_sim_hit_rate.shape == (2, 2):
            sns.heatmap(highest_sim_hit_rate, annot=True, cmap="YlGnBu", fmt=".2f")
            plt.title("Hit Rate of Highest Similarity Path\nSplit by Question Having Any Hits")
            plt.ylabel("Is Highest Similarity Path a Hit")
            plt.xlabel("Does Question Have Any Hit Paths")
            plt.savefig(output_dir / "highest_sim_hit_rate.png", dpi=300, bbox_inches="tight")
    
    # Questions with all unhit paths analysis
    all_unhit_questions = df[df["question_has_hit"] == 0]["question_id"].unique()
    if len(all_unhit_questions) > 0:
        unhit_questions_df = df[df["question_id"].isin(all_unhit_questions)]
        
        plt.figure(figsize=(10, 6))
        sns.histplot(data=unhit_questions_df, x="similarity_score", color="red")
        plt.title("Similarity Score Distribution for Questions with All Unhit Paths")
        plt.xlabel("Similarity Score")
        plt.ylabel("Count")
        plt.savefig(output_dir / "unhit_questions_distribution.png", dpi=300, bbox_inches="tight")
        
        # Compare with hit questions
        hit_questions = df[~df["question_id"].isin(all_unhit_questions)]
        if not hit_questions.empty:
            plt.figure(figsize=(10, 6))
            sns.kdeplot(data=unhit_questions_df, x="similarity_score", label="Unhit Questions", color="red")
            sns.kdeplot(data=hit_questions, x="similarity_score", label="Questions with Hits", color="blue")
            plt.title("Similarity Score Distribution Comparison")
            plt.xlabel("Similarity Score")
            plt.ylabel("Density")
            plt.legend()
            plt.savefig(output_dir / "hit_unhit_questions_comparison.png", dpi=300, bbox_inches="tight")


def main():
    parser = argparse.ArgumentParser(description="Analyze similarity scores and their correlation with correct answers")
    parser.add_argument("--input", type=str, default="./sim_res/path_ppl_scores_path_context_with_sim.jsonl", help="Path to the JSONL file")
    parser.add_argument("--output", type=str, default="./sim_res/analysis_results", help="Output directory for analysis results")
    
    args = parser.parse_args()
    
    # Load and process data
    data = load_jsonl(args.input)
    df = extract_features(data)
    
    # Perform statistical analysis
    results = statistical_analysis(df)
    
    # Print results
    print("\n===== Statistical Analysis Results =====")
    print(f"Number of paths analyzed: {len(df)}")
    print(f"Number of questions: {df['question_id'].nunique()}")
    print(f"Questions with only 1 path: {results['questions_with_one_path']} ({results['questions_with_one_path']/results['total_questions']:.2%})")
    print(f"Questions with all unhit paths: {results['questions_with_all_unhit']} ({results['questions_with_all_unhit']/results['total_questions']:.2%})")
    print(f"Hit rate: {df['is_hit'].mean():.4f}")
    
    print(f"\nMean similarity score for hits: {results['hit_mean']:.4f} ± {results['hit_std']:.4f}")
    print(f"Mean similarity score for non-hits: {results['no_hit_mean']:.4f} ± {results['no_hit_std']:.4f}")
    
    # Only print statistical tests if they were performed
    if not np.isnan(results["p_value"]):
        print(f"\nT-test: t={results['t_stat']:.4f}, p-value={results['p_value']:.8f}")
        print(f"Mann-Whitney U test: U={results['u_stat']:.4f}, p-value={results['u_p_value']:.8f}")
        
        print(f"\nPoint-biserial correlation: r={results['correlation']:.4f}, p-value={results['corr_p_value']:.8f}")
        print(f"Normalized score correlation: r={results['norm_correlation']:.4f}, p-value={results['norm_corr_p_value']:.8f}")
        
        print(f"\nLogistic Regression:")
        print(f"Coefficient: {results['logistic_coef']:.4f}")
        print(f"Intercept: {results['logistic_intercept']:.4f}")
        print(f"Odds Ratio: {results['odds_ratio']:.4f}")
    else:
        print("\nStatistical tests not performed - insufficient data (need both hit and unhit paths)")
    
    print(f"\n===== Question-Level Analysis =====")
    print(f"Questions with at least one hit: {results['questions_with_hits_count']} out of {results['total_questions_count']} ({results['questions_with_hits_count']/results['total_questions_count']:.2%})")
    print(f"Accuracy when taking highest similarity path: {results['highest_sim_accuracy']:.4f}")
    
    if "highest_sim_accuracy_multi" in results and not np.isnan(results["highest_sim_accuracy_multi"]):
        print(f"Accuracy when taking highest similarity path (multi-path questions only): {results['highest_sim_accuracy_multi']:.4f}")
    
    if "hit_questions_highest_sim_accuracy" in results:
        print(f"For questions with hits, highest similarity path accuracy: {results['hit_questions_highest_sim_accuracy']:.4f}")
        
        if "hit_questions_highest_sim_accuracy_multi" in results:
            print(f"For questions with hits (multi-path only), highest similarity path accuracy: {results['hit_questions_highest_sim_accuracy_multi']:.4f}")
        
        print(f"Average rank of hit paths within their questions: {results['avg_hit_path_rank']:.2f}")
    
    # Create visualizations
    plot_results(df, args.output)
    
    # Save full results to JSON
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Convert results to serializable format
    serializable_results = {k: float(v) if isinstance(v, np.float64) else v for k, v in results.items()}
    
    with open(output_dir / "results.json", "w") as f:
        json.dump(serializable_results, f, indent=4)
    
    # Save the DataFrame for further analysis
    df.to_csv(output_dir / "analysis_data.csv", index=False)
    
    print(f"\nAnalysis results and visualizations saved to {args.output}")
    
    # Interpretation of results
    print("\n===== Interpretation =====")
    if not np.isnan(results.get("p_value", float('nan'))):
        if results["p_value"] < 0.05:
            print("The difference in similarity scores between hits and non-hits is statistically significant.")
        else:
            print("The difference in similarity scores between hits and non-hits is NOT statistically significant.")
            
        if results["correlation"] > 0:
            print(f"There is a {'strong' if abs(results['correlation']) > 0.5 else 'moderate' if abs(results['correlation']) > 0.3 else 'weak'} positive correlation between similarity scores and correctness.")
        else:
            print(f"There is a {'strong' if abs(results['correlation']) > 0.5 else 'moderate' if abs(results['correlation']) > 0.3 else 'weak'} negative correlation between similarity scores and correctness.")
        
        if results["odds_ratio"] > 1:
            print(f"For each unit increase in similarity score, the odds of being correct increase by a factor of {results['odds_ratio']:.2f}.")
        else:
            print(f"For each unit increase in similarity score, the odds of being correct decrease by a factor of {1/results['odds_ratio']:.2f}.")
    
    print("\n===== Question-Level Interpretation =====")
    if "hit_questions_highest_sim_accuracy" in results:
        print(f"For questions that have at least one correct path, the highest similarity path is that correct path {results['hit_questions_highest_sim_accuracy']:.2%} of the time.")
    
    # Analyze cross-question similarity issues
    question_avg_scores = df.groupby("question_id")["similarity_score"].mean().reset_index()
    question_has_hit = df.groupby("question_id")["is_hit"].max().reset_index()
    question_stats = pd.merge(question_avg_scores, question_has_hit, on="question_id")
    
    if not question_stats.empty and question_stats["is_hit"].nunique() > 1:
        unhit_q_max_score = question_stats[question_stats["is_hit"] == 0]["similarity_score"].max() if len(question_stats[question_stats["is_hit"] == 0]) > 0 else 0
        hit_q_min_score = question_stats[question_stats["is_hit"] == 1]["similarity_score"].min() if len(question_stats[question_stats["is_hit"] == 1]) > 0 else 1
        
        print(f"\nCross-question similarity analysis:")
        print(f"Maximum average similarity score for questions with NO hits: {unhit_q_max_score:.4f}")
        print(f"Minimum average similarity score for questions with hits: {hit_q_min_score:.4f}")
        
        if unhit_q_max_score > hit_q_min_score:
            print("WARNING: Some questions with no hits have higher average similarity scores than questions with hits.")
            print("This suggests similarity scores may not generalize well across different questions.")
            print("Within-question comparisons are likely more reliable than across-question comparisons.")
        else:
            print("Good news: Questions with hits consistently have higher similarity scores than questions without hits.")
            print("This suggests similarity scores generalize well across different questions.")
    elif results["questions_with_all_unhit"] > 0:
        print("\nNote: All questions either have hits or don't have hits - can't compare across different question types.")


if __name__ == "__main__":
    main()
