import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import holidays
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

class EnergyForecaster:
    def __init__(self):
        self.model = lgb.LGBMRegressor()
        self.feature_cols = None
        self.scaler = StandardScaler()
        
    def load_data(self, train_path, test_path=None):
        """Load and perform initial data analysis."""
        # Load data
        train_df = pd.read_csv(train_path)
        train_df['DateTime'] = pd.to_datetime(train_df['DateTime'])
        
        # Print basic statistics
        print("\nTraining Data Statistics:")
        print(f"Date Range: {train_df['DateTime'].min()} to {train_df['DateTime'].max()}")
        print(f"Number of records: {len(train_df)}")
        print("\nEnergy Consumption Statistics:")
        print(train_df['EnergyConsumption(kWh)'].describe())
        
        # Check for missing values
        missing_values = train_df.isnull().sum()
        if missing_values.sum() > 0:
            print("\nMissing Values:")
            print(missing_values[missing_values > 0])
            train_df.fillna(method='ffill', inplace=True)
        
        # Load test data if provided
        if test_path:
            test_df = pd.read_csv(test_path)
            test_df['DateTime'] = pd.to_datetime(test_df['DateTime'])
            print(f"\nTest Data Range: {test_df['DateTime'].min()} to {test_df['DateTime'].max()}")
        else:
            test_df = None
            
        return train_df, test_df
    
    def plot_energy_patterns(self, df):
        """Create visualizations of energy consumption patterns."""
        plt.figure(figsize=(15, 10))
        
        # Daily pattern
        plt.subplot(2, 2, 1)
        hourly_avg = df.groupby(df['DateTime'].dt.hour)['EnergyConsumption(kWh)'].mean()
        hourly_avg.plot(kind='line', title='Average Energy Consumption by Hour')
        plt.xlabel('Hour of Day')
        plt.ylabel('Average kWh')
        
        # Weekly pattern
        plt.subplot(2, 2, 2)
        weekly_avg = df.groupby(df['DateTime'].dt.dayofweek)['EnergyConsumption(kWh)'].mean()
        weekly_avg.plot(kind='line', title='Average Energy Consumption by Day of Week')
        plt.xlabel('Day of Week')
        plt.ylabel('Average kWh')
        
        plt.tight_layout()
        plt.show()
    
    def feature_engineering(self, df):
        """Perform feature engineering on the dataset."""
        df['hour'] = df['DateTime'].dt.hour
        df['day_of_week'] = df['DateTime'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'] >= 5
        return df
    
    def train_model(self, train_df):
        """Train the model using the training data."""
        train_df = self.feature_engineering(train_df)
        X = train_df[['hour', 'day_of_week', 'is_weekend']]
        y = train_df['EnergyConsumption(kWh)']
        
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.scaler.fit(X_train)
        X_train = self.scaler.transform(X_train)
        X_val = self.scaler.transform(X_val)
        
        self.model.fit(X_train, y_train)
        
        y_pred = self.model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        print(f'Validation RMSE: {rmse}')
    
    def predict(self, test_df):
        """Predict energy consumption for the test data."""
        test_df = self.feature_engineering(test_df)
        X_test = test_df[['hour', 'day_of_week', 'is_weekend']]
        X_test = self.scaler.transform(X_test)
        
        test_df['EnergyConsumption(kWh)'] = self.model.predict(X_test)
        return test_df[['DateTime', 'EnergyConsumption(kWh)']]
    
    def save_predictions(self, predictions, output_path):
        """Save the predictions to a CSV file."""
        predictions.to_csv(output_path, index=False)
        print(f"Predictions saved to {output_path}")

if __name__ == "__main__":
    forecaster = EnergyForecaster()
    train_df, test_df = forecaster.load_data('./data/train_energy.csv', './data/test_energy.csv')
    forecaster.plot_energy_patterns(train_df)
    forecaster.train_model(train_df)
    predictions = forecaster.predict(test_df)
    forecaster.save_predictions(predictions, './result/submission.csv')