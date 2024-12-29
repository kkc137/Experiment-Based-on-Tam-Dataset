from final_model import VaccinePredictionModel

if __name__ == "__main__":
    print("=" * 50)
    print("Vaccine Data Prediction Model Evaluation")
    print("=" * 50)

    # Initialize the VaccinePredictionModel with the data file path
    model = VaccinePredictionModel('2024-10-27 Tam 2016 Data.csv')

    # Run the complete prediction and evaluation pipeline
    results = model.run()

    # Print the final performance evaluation of the models
    print("\nFinal Model Performance Evaluation:")
    print("-" * 30)
    for model_name, metrics in results.items():
        print(f"{model_name} Model:")
        for metric_name, value in metrics.items():
            print(f"- {metric_name}: {value:.3f}")