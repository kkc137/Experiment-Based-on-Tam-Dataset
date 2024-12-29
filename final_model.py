import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import GradientBoostingRegressor
import warnings

warnings.filterwarnings('ignore')



class VaccinePredictionModel:
    """
    This class encapsulates the workflow for loading and processing vaccine binding data,
    training and evaluating time-series prediction models (ARIMA and Gradient Boosting),
    and generating plots for performance comparison and predicted curves.
    """

    def __init__(self, file_path, debug=False):
        """
        Initialize the model with data file path and a debug flag.

        :param file_path: Path to the CSV data file containing the vaccine binding information.
        :param debug: Whether to enable debug output. If True, additional information is printed.
        """
        self.file_path = file_path
        # The following are the target vaccine schemes we want to analyze.
        self.target_schemes = ['Exp-inc', 'Exp-dec', 'Constant', 'Bolus']

        # Placeholders for raw data, processed data, predictions, and metrics.
        self.data = None
        self.processed_data = {}

        # Dictionaries to store model predictions and their corresponding performance metrics.
        self.predictions = {
            'ARIMA': {},
            'GBM': {}
        }
        self.model_metrics = {
            'ARIMA': {'RMSE': [], 'MAE': []},
            'GBM': {'RMSE': [], 'MAE': []}
        }

        # Flag to toggle debug prints.
        self.debug = debug

        # Use a specific style for plotting.
        plt.style.use('ggplot')
        print("\n[Start] Initialization completed")

    def preprocess_time_series(self, series, scheme_name="Unknown"):
        """
        Preprocess a time series by clipping outliers using the 3-sigma rule.

        :param series: 1D array or list of data points (binding values).
        :param scheme_name: Name of the vaccine scheme for debugging/logging.
        :return: Numpy array of the clipped time series data.
        """
        ts = pd.Series(series)
        n_before = len(ts)

        # Calculate mean and standard deviation for outlier clipping.
        mean_ = ts.mean()
        std_ = ts.std()

        # Clip the values that fall outside the [mean - 3*std, mean + 3*std] range.
        ts_clipped = ts.clip(lower=mean_ - 3 * std_, upper=mean_ + 3 * std_)

        if self.debug:
            print(f"\n>>> [DEBUG] Scheme={scheme_name}, Original data points={n_before}")
            print("    Original data (first 5):", list(np.round(ts.values[:5], 3)), "...")
            print("    After clipping (first 5):", list(np.round(ts_clipped.values[:5], 3)), "...")

        # Return the clipped time series as a numpy array.
        return ts_clipped.values

    def load_and_process_data(self):
        """
        Load the dataset from the specified CSV file, filter it by the target schemes,
        remove invalid entries, and preprocess it for each vaccine scheme.
        """
        print("\n[1/5] Data Loading and Preprocessing Phase")

        # Load the CSV file into a Pandas DataFrame.
        self.data = pd.read_csv(self.file_path)

        # Ensure 'Binding' column is numeric, coerce invalid values to NaN.
        self.data['Binding'] = pd.to_numeric(self.data['Binding'], errors='coerce')

        # Filter the data to only include target schemes, and remove rows with NaN in 'Binding'.
        self.data = self.data[self.data['Vaccine Scheme'].isin(self.target_schemes)]
        self.data = self.data.dropna(subset=['Binding'])

        # Convert the 'Time' column (e.g., "Day10") to an integer representing the day.
        self.data['Time_Numeric'] = self.data['Time'].str.replace('Day', '').astype(int)

        print(f"- Dataset loaded: {len(self.data)} records")
        print("- Starting data preprocessing...")

        # Group each target scheme by day and compute the mean binding value for that day.
        for scheme in self.target_schemes:
            scheme_data = self.data[self.data['Vaccine Scheme'] == scheme]
            grouped = scheme_data.groupby('Time_Numeric')['Binding'].mean().reset_index()

            # Sort grouped data in ascending order by 'Time_Numeric'.
            grouped.sort_values('Time_Numeric', inplace=True)
            values = grouped['Binding'].values.astype(float)

            if self.debug:
                print(f"\n>>> [DEBUG] Scheme={scheme}, Grouped into {len(values)} data points:")
                print(grouped.head(10))

            # Preprocess (e.g., outlier clipping) and store in `processed_data`.
            self.processed_data[scheme] = self.preprocess_time_series(
                values, scheme_name=scheme
            )
        print("- Data preprocessing completed")

    def optimize_arima_order(self, series):
        """
        Attempt to find the best (p, d, q) order for an ARIMA model by minimizing the AIC score.

        :param series: 1D numpy array of training data.
        :return: Tuple (p, d, q) corresponding to the best ARIMA order found.
        """
        best_aic = float('inf')
        best_order = None

        # We will do a grid search over a small range of p, d, q values.
        for p in range(5):
            for d in range(3):
                for q in range(5):
                    try:
                        # Create and fit an ARIMA model with a given (p, d, q) order.
                        model = ARIMA(series, order=(p, d, q))
                        results = model.fit()

                        # Update best order if the AIC is improved.
                        if results.aic < best_aic:
                            best_aic = results.aic
                            best_order = (p, d, q)
                    except:
                        # Some combinations of p, d, q may fail to converge or be invalid.
                        continue

        # Return the best order found.
        return best_order

    def train_and_evaluate(self):
        """
        Split the processed data into training and test sets, train models (ARIMA, GBM),
        evaluate performance (RMSE, MAE), and store the predictions and metrics.
        """
        print("\n[2/5] Model Training and Evaluation Phase")
        print("- Starting model training...")

        # We hold out the last 4 data points as a test set.
        test_size = 4

        # Iterate over each scheme's processed data.
        for scheme, series_1d in self.processed_data.items():
            total_len = len(series_1d)
            train_size = total_len - test_size

            # If we don't have enough data to create a train/test split, skip.
            if train_size <= 0:
                print(f"!!! Warning: {scheme} data is insufficient for training/testing!")
                continue

            # Partition the data into train and test segments.
            train_data = series_1d[:train_size]
            test_data = series_1d[train_size:]

            if self.debug:
                print(f"\n>>> [DEBUG] {scheme}: total_len={total_len}, train_size={train_size}, test_size={test_size}")
                print("    Train data (first 5):", np.round(train_data[:5], 3), "...")
                print("    Test data:", np.round(test_data, 3))

            # ========== ARIMA ==========
            try:
                # Find the best ARIMA order based on the training data.
                best_order = self.optimize_arima_order(train_data)
                # Train the ARIMA model with the best order.
                arima_model = ARIMA(train_data, order=best_order).fit()

                # Forecast for the test period + 14 extra data points (e.g., future predictions).
                steps = test_size + 14
                arima_pred = arima_model.forecast(steps)

                # Store predictions in the dictionary.
                self.predictions['ARIMA'][scheme] = arima_pred

                # Calculate evaluation metrics (RMSE, MAE) on the test portion of the forecast.
                rmse = np.sqrt(mean_squared_error(test_data, arima_pred[:test_size]))
                mae = mean_absolute_error(test_data, arima_pred[:test_size])
                self.model_metrics['ARIMA']['RMSE'].append(rmse)
                self.model_metrics['ARIMA']['MAE'].append(mae)

                if self.debug:
                    print(f"    [ARIMA] Best order={best_order}, First few predictions: {np.round(arima_pred[:5], 3)}")
            except Exception as e:
                print(f"ARIMA model training failed ({scheme}): {e}")

            # ========== GBM (Gradient Boosting) ==========
            try:
                # Prepare training sequences using a sliding window approach of size 5.
                X, y = [], []
                window_size = 5
                for i in range(len(train_data) - window_size):
                    X.append(train_data[i:i + window_size])
                    y.append(train_data[i + window_size])
                X = np.array(X)
                y = np.array(y)

                # If there's no data left after the sliding window, skip training.
                if len(X) == 0:
                    print(f"!!! Warning: {scheme} data is insufficient for GBM training.")
                    continue

                # Initialize and train the Gradient Boosting Regressor.
                gbm = GradientBoostingRegressor(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=3,
                    random_state=42
                )
                gbm.fit(X, y)

                # Perform forecasting by rolling the window forward one step at a time.
                gbm_pred = []

                # Start with the last 'window_size' points from the training data.
                current_sequence = train_data[-window_size:].copy()
                for _ in range(test_size + 14):
                    pred_val = gbm.predict(current_sequence.reshape(1, -1))
                    gbm_pred.append(pred_val[0])

                    # Shift the window by one and add the new prediction at the end.
                    current_sequence = np.roll(current_sequence, -1)
                    current_sequence[-1] = pred_val[0]

                # Convert predictions to a numpy array and store.
                self.predictions['GBM'][scheme] = np.array(gbm_pred)

                # Evaluate GBM on the test portion of the prediction.
                rmse = np.sqrt(mean_squared_error(test_data, gbm_pred[:test_size]))
                mae = mean_absolute_error(test_data, gbm_pred[:test_size])
                self.model_metrics['GBM']['RMSE'].append(rmse)
                self.model_metrics['GBM']['MAE'].append(mae)

                if self.debug:
                    print(f"    [GBM] First few predictions: {np.round(gbm_pred[:5], 3)}")
            except Exception as e:
                print(f"GBM model training failed ({scheme}): {e}")

        print("- Model training completed")

    def calculate_overall_performance(self):
        """
        Calculate overall performance metrics across all target schemes for each model.

        :return: A dictionary containing the average RMSE and MAE for ARIMA and GBM.
        """
        print("\n[3/5] Model Performance Calculation Phase")
        overall_performance = {}

        # Loop through each model and each metric to compute their mean values.
        for model in self.model_metrics:
            overall_performance[model] = {}
            for metric in self.model_metrics[model]:
                vals = self.model_metrics[model][metric]
                # If we have no values, store NaN. Otherwise, store the mean.
                overall_performance[model][metric] = np.mean(vals) if len(vals) > 0 else np.nan

        print("- Performance metrics calculated")
        return overall_performance

    def plot_performance_comparison(self, overall_performance):
        """
        Plot a side-by-side bar chart comparing RMSE and MAE for ARIMA and GBM with improved visualization.

        :param overall_performance: A dictionary of overall metrics for each model.
        """
        print("\n[4/5] Performance Comparison Visualization")

        # Define the metrics to compare
        metrics = ['RMSE', 'MAE']
        fig, ax = plt.subplots(figsize=(12, 7))  # Increased figure size for better readability

        # Define x positions for bars
        x = np.arange(len(metrics))
        width = 0.35  # Width of the bars

        # Extract metric values for ARIMA and GBM
        arima_vals = [overall_performance['ARIMA'][m] for m in metrics]
        gbm_vals = [overall_performance['GBM'][m] for m in metrics]

        # Plot bars for ARIMA and GBM with distinct colors
        rects1 = ax.bar(x - width / 2, arima_vals, width, label='ARIMA', color='#1f77b4', alpha=0.9)
        rects2 = ax.bar(x + width / 2, gbm_vals, width, label='GBM', color='#ff7f0e', alpha=0.9)

        # Add annotations to display values on the bars
        for rects in [rects1, rects2]:
            for rect in rects:
                height = rect.get_height()
                if not np.isnan(height):  # Skip annotation for NaN values
                    ax.text(
                        rect.get_x() + rect.get_width() / 2,  # Position in the center of the bar
                        height + 0.02,  # Slightly above the bar
                        f'{height:.2f}',  # Format to two decimal places
                        ha='center', va='bottom', fontsize=10, fontweight='bold', color='black'
                    )

        # Set labels and title
        ax.set_ylabel('Score', fontsize=14)
        ax.set_title('Model Performance Comparison', fontsize=16, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, fontsize=12)
        ax.legend(fontsize=12, loc='upper right')

        # Add grid lines for better visual guidance
        ax.grid(axis='y', linestyle='--', alpha=0.6)

        # Adjust layout and save the plot
        plt.tight_layout(pad=2.0)
        plt.savefig('model_performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("- Performance comparison plot generated")

    def plot_prediction_curves(self):
        """
        For each model, plot the measured data points for each scheme along with
        the predicted curves (including some forecast into the future).
        """
        print("\n[5/5] Prediction Results Visualization")

        # Define colors for each scheme for consistent plotting.
        colors = {
            'Exp-inc': '#1f77b4',
            'Exp-dec': '#2ca02c',
            'Constant': '#ff7f0e',
            'Bolus': '#d62728'
        }

        # We'll plot predictions for each model separately.
        for model_name in ['ARIMA', 'GBM']:
            plt.figure(figsize=(15, 8))

            # Loop over each vaccine scheme and plot measured vs. predicted values.
            for scheme in self.target_schemes:
                # If there's no prediction available for this scheme/model, skip.
                if scheme not in self.predictions[model_name]:
                    continue

                # Extract the measured data for this scheme.
                scheme_data = self.data[self.data['Vaccine Scheme'] == scheme]
                if len(scheme_data) == 0:
                    continue

                # Group the data by time and get the mean binding value for each day.
                measured_df = scheme_data.groupby('Time_Numeric')['Binding'].mean().reset_index()
                measured_days = measured_df['Time_Numeric'].values
                measured_values = measured_df['Binding'].values

                # Retrieve the model predictions for this scheme.
                pred_vals = self.predictions[model_name][scheme]

                # We'll assume the last measured day is the "starting point"
                # and then forecast weekly steps for however many points we predicted.
                last_day = measured_days[-1]
                future_days = [last_day + 7 * (i + 1) for i in range(len(pred_vals))]

                # Plot the measured data in a solid line.
                plt.plot(measured_days, measured_values,
                         '-',
                         color=colors.get(scheme, 'blue'),
                         label=f'{scheme} (Measured)',
                         linewidth=2)

                # Plot the predictions in a dashed line.
                plt.plot(future_days, pred_vals,
                         '--',
                         color=colors.get(scheme, 'orange'),
                         label=f'{scheme} ({model_name} Prediction)',
                         linewidth=2)

                # Optionally connect the last measured point to the first predicted point.
                plt.plot([measured_days[-1], future_days[0]],
                         [measured_values[-1], pred_vals[0]],
                         ':',
                         color=colors.get(scheme, 'gray'),
                         alpha=0.7)

            # Configure and save the figure for this model.
            plt.title(f'{model_name} Model - Vaccine Schemes Prediction (No Scaling)')
            plt.xlabel('Days')
            plt.ylabel('Binding Value')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f'{model_name.lower()}_predictions_with_measured.png', dpi=300, bbox_inches='tight')
            plt.close()

        print("- Prediction curves generated")

    def run(self):
        """
        Run the full workflow:
          1) Load and process data,
          2) Train and evaluate models,
          3) Calculate overall performance,
          4) Plot performance comparison,
          5) Plot prediction curves,
          6) Return overall performance metrics.
        """
        self.load_and_process_data()
        self.train_and_evaluate()
        overall_perf = self.calculate_overall_performance()
        self.plot_performance_comparison(overall_perf)
        self.plot_prediction_curves()

        print("\n[Completed] Prediction and Evaluation Workflow Finished")
        return overall_perf
