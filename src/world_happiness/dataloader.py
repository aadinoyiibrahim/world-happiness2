import pandas as pd
import os
import logging
from typing import List
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataLoader:
    """
    Class for loading and preprocessing World Happiness Report data.
    Demonstrates OOP in software engineering.
    """
    def __init__(self, data_dir: str = 'data/'):
        self.data_dir = data_dir
        self.years: List[int] = [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]
        self.df: pd.DataFrame = pd.DataFrame()

    def load_data(self) -> pd.DataFrame:
        """
        Load and combine CSV files for multiple years, handling schema differences.
        :return: Combined DataFrame with 'Year' column.
        """
        dfs = []
        for year in self.years:
            file_path = os.path.join(self.data_dir, f"{year}.csv")
            if os.path.exists(file_path):
                df_year = pd.read_csv(file_path)
                # Standardize column names
                rename_dict = {
                    'Country name': 'Country',
                    'Country or region': 'Country',
                    'Happiness.Rank': 'Happiness Rank',
                    'Happiness.Score': 'Happiness Score',
                    'Ladder score': 'Happiness Score',
                    'Score': 'Happiness Score',
                    'Economy (GDP per Capita)': 'GDP per capita',
                    'Economy..GDP.per.Capita.': 'GDP per capita',
                    'Logged GDP per capita': 'GDP per capita',
                    'Family': 'Social support',
                    'Social support': 'Social support',
                    'Health (Life Expectancy)': 'Healthy life expectancy',
                    'Health..Life.Expectancy.': 'Healthy life expectancy',
                    'Healthy life expectancy': 'Healthy life expectancy',
                    'Freedom': 'Freedom to make life choices',
                    'Freedom to make life choices': 'Freedom to make life choices',
                    'Trust (Government Corruption)': 'Perceptions of corruption',
                    'Trust..Government.Corruption.': 'Perceptions of corruption',
                    'Perceptions of corruption': 'Perceptions of corruption',
                    'Generosity': 'Generosity',
                    'Dystopia Residual': 'Dystopia + residual',
                    'Dystopia.Residual': 'Dystopia + residual'
                }
                df_year.rename(columns=rename_dict, inplace=True)
                df_year['Year'] = year
                dfs.append(df_year)
            else:
                logger.warning(f"File for {year} not found.")
        
        if dfs:
            self.df = pd.concat(dfs, ignore_index=True)
            # Handle missing values with median imputation
            self.df.fillna(self.df.median(numeric_only=True), inplace=True)
            logger.info(f"Loaded data for {len(dfs)} years with shape {self.df.shape}")
        return self.df

    def preprocess(self) -> pd.DataFrame:
        """
        Scale features and encode categoricals.
        :return: Preprocessed DataFrame.
        """
        if self.df.empty:
            raise ValueError("Data not loaded. Call load_data() first.")
        scaler = StandardScaler()
        numeric_cols = self.df.select_dtypes(include=['float64', 'int64']).columns.drop('Year', errors='ignore')
        self.df[numeric_cols] = scaler.fit_transform(self.df[numeric_cols])
        return self.df