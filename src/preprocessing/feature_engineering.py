from src.utils.logger import get_logger

logger = get_logger(__name__)

def create_features(df):

    logger.info("Creating engineered features")

    if "work_hours_per_day" in df.columns and "sleep_hours" in df.columns:

        df["work_life_balance"] = df["work_hours_per_day"] / (df["sleep_hours"] + 1)

    if "number_of_notifications" in df.columns and "daily_social_media_time" in df.columns:

        df["digital_distraction_score"] = (
            df["number_of_notifications"] * df["daily_social_media_time"]
        )

    logger.info("Feature engineering completed")

    return df