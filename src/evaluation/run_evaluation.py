from src.evaluation.evaluate_models import evaluate_models
from src.evaluation.explain_model import feature_importance
from src.evaluation.generate_insights import generate_insights

def run_evaluation():

    evaluate_models()
    feature_importance()
    generate_insights()

    print("Evaluation phase completed")