import pandas as pd
from rf_pipeline import RandomForestPipeline
import time


def main():
    """
    Main function to run Random Forest pipeline.
    """
    print("Starting the Random Forest pipeline...")

    # Load data
    df = pd.read_csv("immo-ml-data.csv")

    # Define features and target
    X = df.drop(columns=["Price"])  # Features
    y = df["Price"]  # Target

    # Initialize pipeline
    pipeline = RandomForestPipeline(degree=2, max_depth=10)

    # Train model + timing
    print("Model is training!")
    print("It's the eye of the tiger, it's the thrill of the fight...")
    start_time = time.time()
    pipeline.train(X, y)
    end_time = time.time()

    # Calculate elapsed time
    elapsed_time = end_time - start_time
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    print(f"Model training time: {minutes} minutes and {seconds} seconds")

    # Evaluate model
    metrics = pipeline.evaluate()
    print(
        f"Model Evaluation:\n"
        f"Training - RMSE: {metrics['training']['RMSE']:.2f}, MAE: {metrics['training']['MAE']:.2f}, "
        f"R²: {metrics['training']['R²']:.2f}, MAPE: {metrics['training']['MAPE']:.2f}%, "
        f"sMAPE: {metrics['training']['sMAPE']:.2f}%\n"
        f"Test - RMSE: {metrics['test']['RMSE']:.2f}, MAE: {metrics['test']['MAE']:.2f}, "
        f"R²: {metrics['test']['R²']:.2f}, MAPE: {metrics['test']['MAPE']:.2f}%, "
        f"sMAPE: {metrics['test']['sMAPE']:.2f}%"
    )

    # Save predictions to csv file
    print("\nSaving predictions...")
    pipeline.save_predictions(file_name="model_predictions.csv")

    # Analyze feature importance
    feature_importances = pipeline.feature_importance()
    feature_importances.to_csv("feature_importances.csv", index=False)
    print("\nFeature Importances saved to feature_importances.csv")


if __name__ == "__main__":
    main()
