from src.evaluation.evaluate_models import evaluate_models
from src.evaluation.feature_importance import run_feature_importance
from src.evaluation.shap_analysis import run_shap_analysis
from src.evaluation.generate_insights import generate_insights


def run_evaluation():

    evaluate_models()

    run_feature_importance()

    run_shap_analysis()

    generate_insights()

    print("Phase 5 completed")