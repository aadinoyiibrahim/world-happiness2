import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima.model import ARIMA
import optuna
import shap
import streamlit as st
import logging

logger = logging.getLogger(__name__)

class HappinessModel:
    """
    Advanced modeling class for happiness prediction.
    """
    def __init__(self, df: pd.DataFrame):
        self.df = df
        feature_cols = ['GDP per capita', 'Social support', 'Healthy life expectancy',
                        'Freedom to make life choices', 'Generosity', 'Perceptions of corruption']
        self.X = self.df[feature_cols]
        self.y = self.df['Happiness Score']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )

    def stack_models(self) -> float:
        """
        Stacking ensemble: RandomForest + Neural Net.
        :return: R2 score.
        """
        # Base model: RandomForest
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(self.X_train, self.y_train)
        
        # Neural Net with Torch
        class Net(nn.Module):
            def __init__(self, input_size):
                super(Net, self).__init__()
                self.fc1 = nn.Linear(input_size, 64)
                self.fc2 = nn.Linear(64, 32)
                self.fc3 = nn.Linear(32, 1)
            
            def forward(self, x):
                x = torch.relu(self.fc1(x))
                x = torch.relu(self.fc2(x))
                return self.fc3(x)
        
        net = Net(input_size=self.X.shape[1])
        criterion = nn.MSELoss()
        optimizer = optim.Adam(net.parameters(), lr=0.001)
        train_dataset = TensorDataset(
            torch.tensor(self.X_train.values, dtype=torch.float32),
            torch.tensor(self.y_train.values, dtype=torch.float32).unsqueeze(1)
        )
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        
        for epoch in range(100):
            for data, target in train_loader:
                optimizer.zero_grad()
                output = net(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
        
        # Stacking: Use predictions as meta-features
        rf_preds_train = rf.predict(self.X_train)
        nn_preds_train = net(torch.tensor(self.X_train.values, dtype=torch.float32)).detach().numpy().flatten()
        meta_train = np.column_stack((rf_preds_train, nn_preds_train))
        
        rf_preds_test = rf.predict(self.X_test)
        nn_preds_test = net(torch.tensor(self.X_test.values, dtype=torch.float32)).detach().numpy().flatten()
        meta_test = np.column_stack((rf_preds_test, nn_preds_test))
        
        # Meta model: Linear Regression
        meta_model = LinearRegression()
        meta_model.fit(meta_train, self.y_train)
        stacked_preds = meta_model.predict(meta_test)
        
        r2 = r2_score(self.y_test, stacked_preds)
        logger.info(f"Stacked model R2: {r2}")
        return r2

    def hyperparam_optimization(self, trial: optuna.Trial) -> float:
        """
        Optuna objective for hyperparam tuning (example on RF).
        """
        n_estimators = trial.suggest_int('n_estimators', 50, 200)
        max_depth = trial.suggest_int('max_depth', 5, 20)
        rf = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        return cross_val_score(rf, self.X, self.y, cv=5, scoring='r2').mean()

    def run_optuna(self, n_trials: int = 20):
        study = optuna.create_study(direction='maximize')
        study.optimize(self.hyperparam_optimization, n_trials=n_trials)
        logger.info(f"Best params: {study.best_params}")

    def time_series_forecast(self, country: str = 'Finland'):
        """
        ARIMA for forecasting happiness for a specific country.
        """
        df_country = self.df[self.df['Country'] == country].sort_values('Year')
        if len(df_country) < 3:
            raise ValueError(f"Insufficient data for {country}.")
        model = ARIMA(df_country['Happiness Score'], order=(1,1,1))
        results = model.fit()
        forecast = results.forecast(steps=5)
        print(f"Forecasted Happiness for {country} (next 5 years): {forecast}")

    def explainable_ai(self, model):
        """
        SHAP for explainability on a tree-based model.
        """
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(self.X_test)
        shap.summary_plot(shap_values, self.X_test, show=False)
        plt.savefig('shap_summary.png')
        plt.show()

def streamlit_app(df: pd.DataFrame):
    """
    Simple Streamlit app for predictions. Run with `streamlit run models.py`.
    """
    st.title('World Happiness Predictor')
    # Sliders for features
    gdp = st.slider('GDP per capita', -3.0, 3.0, 0.0)  # Scaled
    support = st.slider('Social support', -3.0, 3.0, 0.0)
    health = st.slider('Healthy life expectancy', -3.0, 3.0, 0.0)
    freedom = st.slider('Freedom to make life choices', -3.0, 3.0, 0.0)
    generosity = st.slider('Generosity', -3.0, 3.0, 0.0)
    corruption = st.slider('Perceptions of corruption', -3.0, 3.0, 0.0)
    
    inputs = np.array([[gdp, support, health, freedom, generosity, corruption]])
    # Placeholder model (train a RF)
    rf = RandomForestRegressor(random_state=42).fit(df.iloc[:, :-1].select_dtypes(include=['float64']), df['Happiness Score'])
    pred = rf.predict(inputs)[0]
    st.write(f"Predicted Happiness Score: {pred}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'app':
        # Example: Load data and run app
        from dataloader import DataLoader
        loader = DataLoader()
        df = loader.load_data()
        df = loader.preprocess()
        streamlit_app(df)