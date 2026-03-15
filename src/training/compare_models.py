import pandas as pd


def compare_models(results):

    df = pd.DataFrame(results)

    df = df.sort_values("rmse")

    best_model = df.iloc[0]

    return df, best_model