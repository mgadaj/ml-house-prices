# database
DATA_NAME = "houseprices.csv"
TARGET = "SalePrice"

# model
MODEL_NAME = "lasso_regression"

# variables
SELECTED_VARIABLES = ["MSSubClass", "MSZoning", "Neighborhood", "OverallQual",
                      "OverallCond", "YearRemodAdd", "RoofStyle", "MasVnrType",
                      "BsmtQual", "BsmtExposure", "HeatingQC", "CentralAir",
                      "1tFlrSF", "GrLivArea", "BsmtFullBath", "KitchenQual",
                      "Fireplaces", "FireplaceQu", "GarageType", "GarageFinish",
                      "GarageCars", "PavedDrive", "YrSold"]

# categorical variables with nan
CATEGORICAL_VARIABLES_WITH_NAN = ["MasVnrType", "BsmtQual", "BsmtExposure",
                                  "Fireplaces", "GarageType", "GarageFinish"]
# numerical variables with nan
NUMERICAL_VARIABLES_WITH_NAN = ["LotFrontage"]

# variables to drop
DROP_VARIABLES = "YrSold"

# temporal variables
TEMPORAL_VARIABLES = "YearRemodAdd"

# categorical variables to encode
CATEGORICAL_VARIABLES_ENCODE = ["MSZoning", "Neighborhood", "RoofStyle",
                                "MasVnrType", "BsmtQual", "BsmtExposure",
                                "HeatingQC", "CentralAir", "KitchenQual",
                                "FireplaceQu", "GarageType", "GarageFinish",
                                "PavedDrive"]

# variables to logarithm transformation
NUMERICAL_VARIABLES_LOGARITHM_TRANSFORM = ["LotFrontage", "1tFlrSF", "GrLivArea"]
