import pytest
from world_happiness.dataloader import DataLoader
import pandas as pd
import os

@pytest.fixture
def mock_data(tmp_path):
    # Create mock CSV
    d = tmp_path / "data"
    d.mkdir()
    path = d / "2015.csv"
    df = pd.DataFrame({'Country': ['Test'], 'Happiness Score': [5.0]})
    df.to_csv(path, index=False)
    return str(d)

def test_load_data(mock_data):
    loader = DataLoader(data_dir=mock_data)
    loader.years = [2015]  # Mock
    df = loader.load_data()
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert 'Year' in df.columns

def test_preprocess(mock_data):
    loader = DataLoader(data_dir=mock_data)
    loader.years = [2015]
    loader.load_data()
    df = loader.preprocess()
    assert df['Happiness Score'].mean() == pytest.approx(0, abs=1e-5)  # Scaled