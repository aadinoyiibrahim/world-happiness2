import pytest
from world_happiness.eda import perform_eda
import pandas as pd

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'Year': [2020, 2021],
        'Happiness Score': [5.0, 6.0],
        'GDP per capita': [1.0, 1.2],
        'Country': ['Test1', 'Test2']
    })

def test_perform_eda(sample_df, capsys):
    perform_eda(sample_df)
    captured = capsys.readouterr()
    assert "anomalies" in captured.out  # Checks print statements