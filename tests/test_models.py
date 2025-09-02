import pytest
from world_happiness.models import HappinessModel
import pandas as pd
import numpy as np

@pytest.fixture
def sample_df():
    data = {
        'GDP per capita': np.random.rand(10),
        'Social support': np.random.rand(10),
        'Healthy life expectancy': np.random.rand(10),
        'Freedom to make life choices': np.random.rand(10),
        'Generosity': np.random.rand(10),
        'Perceptions of corruption': np.random.rand(10),
        'Happiness Score': np.random.rand(10),
        'Country': ['Test']*10,
        'Year': [2020]*10
    }
    return pd.DataFrame(data)

def test_stack_models(sample_df):
    model = HappinessModel(sample_df)
    r2 = model.stack_models()
    assert -1 <= r2 <= 1  # Valid R2 range

def test_time_series_forecast(sample_df):
    model = HappinessModel(sample_df)
    with pytest.raises(ValueError):  # Insufficient data for forecast
        model.time_series_forecast()