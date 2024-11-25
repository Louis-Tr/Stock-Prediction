import pandas as pd


class StockModel:
    """
    Comprehensive class for end-to-end machine learning workflow.
    Includes data fetching, preprocessing, normalization, train-test splitting,
    model training, evaluation, and saving/loading the trained model.
    """

    def __init__(self):
        self.data = None
        self.train_data = None
        self.test_data = None
        self.model = None
        self.evaluation = None

    def fetch_data(self, data):
        """
        Fetches the data to the model.

        Args:
            data (pd.DataFrame): Data to be fetched.

        Returns:
            pd.DataFrame: The fetched data stored in the model.

        Raises:
            AssertionError: If the operation fails due to invalid input or unexpected error.
        """
        try:
            # Attempt to fetch and store the data
            self.data = pd.DataFrame(data)
            assert not self.data.empty, "The data is empty after fetching."
            return self.data
        except Exception as e:
            # Raise an assertion error with details if fetching fails
            raise AssertionError(f"Failed to fetch data: {e}")

    def show_headings(self):
        """
        Displays the first 5 rows of the fetched data.

        Raises:
            AssertionError: If the data is not loaded or any unexpected error occurs.
        """
        try:
            # Ensure data is loaded before attempting to display it
            assert self.data is not None, "No data has been loaded. Fetch the data first."
            print(self.data.head())
        except Exception as e:
            raise AssertionError(f"Failed to show data: {e}")

    def clean_data(self):
        """
        Cleans and preprocesses the fetched data.
        Removes NaN values and unnecessary columns if needed.
        """
        if self.data is not None:
            self.data.dropna(inplace=True)
        else:
            raise ValueError("No data to clean. Please fetch data first.")

    def normalize_data(self):
        """
        Normalizes the data for machine learning models.
        Uses Min-Max Scaling as the default normalization technique.

        Raises:
            ValueError: If no data is available for normalization.
        """
        if self.data is not None:
            try:
                from sklearn.preprocessing import MinMaxScaler

                # Apply Min-Max Scaling
                scaler = MinMaxScaler()
                numeric_columns = self.data.select_dtypes(include=["number"]).columns
                self.data[numeric_columns] = scaler.fit_transform(self.data[numeric_columns])
                print("Data has been normalized successfully.")
            except Exception as e:
                raise ValueError(f"Failed to normalize data: {e}")
        else:
            raise ValueError("No data to normalize. Please fetch data first.")

    def split_data(self, test_size=0.2):
        """
        Splits the data into training and testing sets.

        Args:
            test_size (float): Proportion of the dataset to include in the test split.

        Returns:
            tuple: Training and testing data.
        """
        from sklearn.model_selection import train_test_split

        if self.data is not None:
            X = self.data.drop(columns=["High", "Low"])
            y = self.data[["High", "Low"]]
            self.train_data, self.test_data = train_test_split(
                (X, y), test_size=test_size, random_state=42
            )
        else:
            raise ValueError("No data to split. Please fetch and preprocess data first.")

    def assign_model(self, model):
        """
        Assigns a machine learning model to the class.

        Args:
            model: An instance of a machine learning model.
        """
        try:
            self.model = model
        except Exception as e:
            raise ValueError(f"Failed to assign model: {e}")

    def train_model(self):
        """
        Trains using assigned model and split data.
        """
        try:
            if self.model is not None:
                if self.train_data is not None:
                    X_train, y_train = self.train_data
                    self.model.fit(X_train, y_train)
                    print("Model trained successfully.")
                else:
                    raise ValueError("No training data available.")
            else:
                raise ValueError("No model assigned for training.")
        except Exception as e:
            raise ValueError(f"Failed to train model: {e}")

    def evaluate_model(self):
        """
        Evaluates the trained model using testing data.

        Returns:
            dict: Evaluation metrics.
        """
        try:
            if self.model is not None:
                if self.test_data is not None:
                    X_test, y_test = self.test_data
                    predictions = self.model.predict(X_test)
                    from sklearn.metrics import mean_absolute_error, mean_squared_error

                    self.evaluation = {
                        "MAE": mean_absolute_error(y_test, predictions),
                        "MSE": mean_squared_error(y_test, predictions),
                        "RMSE": mean_squared_error(y_test, predictions, squared=False),
                    }
                else:
                    raise ValueError("No test data available for evaluation.")
            else:
                raise ValueError("No model available for evaluation.")

            return self.evaluation
        except Exception as e:
            raise ValueError(f"Failed to evaluate model: {e}")


def save_model(self, file_path):
    """
    Saves the trained model to a file using joblib.

    Args:
        file_path (str): File path to save the model.
    """
    import joblib

    if self.model is not None:
        joblib.dump(self.model, file_path)
    else:
        raise ValueError("No model to save. Train the model first.")


def load_model(self, file_path):
    """
    Loads a trained model from a file.

    Args:
        file_path (str): File path to load the model.
    """
    import joblib

    self.model = joblib.load(file_path)
