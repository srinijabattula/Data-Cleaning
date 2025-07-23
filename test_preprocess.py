import unittest
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA

class TestPreprocessing(unittest.TestCase):
    
    def setUp(self):
        # Create a sample DataFrame with missing values and numeric/categorical columns
        self.df = pd.DataFrame({
            'A': [1, 2, np.nan, 4],
            'B': [5, np.nan, 7, 8],
            'Category': ['X', 'Y', 'X', 'Z']
        })

    def test_missing_values_handling(self):
        imputer = SimpleImputer(strategy='mean')
        df_imputed = self.df.copy()
        numeric_cols = df_imputed.select_dtypes(include=['float64', 'int64']).columns
        df_imputed[numeric_cols] = imputer.fit_transform(df_imputed[numeric_cols])
        self.assertFalse(df_imputed[numeric_cols].isnull().any().any(), "Missing values not handled properly.")

    def test_standard_scaling(self):
        df_scaled = self.df.copy()
        df_scaled[['A', 'B']] = df_scaled[['A', 'B']].fillna(df_scaled[['A', 'B']].mean())
        scaler = StandardScaler()
        df_scaled[['A', 'B']] = scaler.fit_transform(df_scaled[['A', 'B']])
        self.assertAlmostEqual(df_scaled['A'].mean(), 0, delta=1e-6)
        self.assertAlmostEqual(df_scaled['B'].mean(), 0, delta=1e-6)

    def test_feature_engineering(self):
        df_feat = self.df.copy()
        df_feat['A'] = df_feat['A'].fillna(df_feat['A'].mean())
        df_feat['B'] = df_feat['B'].fillna(df_feat['B'].mean())
        df_feat['combined'] = df_feat['A'] * df_feat['B']
        self.assertIn('combined', df_feat.columns, "Feature engineering failed.")

    def test_pca_application(self):
        df_pca = self.df.copy()
        df_pca[['A', 'B']] = df_pca[['A', 'B']].fillna(df_pca[['A', 'B']].mean())
        pca = PCA(n_components=1)
        result = pca.fit_transform(df_pca[['A', 'B']])
        self.assertEqual(result.shape, (4, 1), "PCA output shape mismatch.")

if __name__ == '__main__':
    unittest.main()

