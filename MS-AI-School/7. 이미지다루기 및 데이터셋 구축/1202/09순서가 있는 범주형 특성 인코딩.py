import numpy as np
from sklearn.preprocessing import OrdinalEncoder

features = np.array([["Low", 10], ["High", 50], ["Medium", 3]])
ordinal_encoding = OrdinalEncoder()
ordinal_encoding.fit_transform(features)
ordinal_encoding_data = ordinal_encoding.categories_

print("ordinal_encoding.categories_", ordinal_encoding_data)
