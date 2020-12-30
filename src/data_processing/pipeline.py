from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Lasso
import processing as ps
import config as cfg

model_pipeline = Pipeline([
    ("imputer_numerical_variable",
     ps.ImputerNumericalVariable(variables=cfg.NUMERICAL_VARIABLES_WITH_NAN)),
    ("imputer_categorical_variable",
     ps.ImputerCategoricalVariable(variables=cfg.CATEGORICAL_VARIABLES_WITH_NAN)),
    ("processor_temporal_variable",
     ps.ProcessorTemporalVariable(variables=cfg.TEMPORAL_VARIABLES, related_variable=cfg.DROP_VARIABLES)),
    ("encoder_rare_label",
     ps.EncoderRareLabel(tolerance=0.01, variables=cfg.CATEGORICAL_VARIABLES_ENCODE)),
    ("encoder_categorical_variable",
     ps.EncoderCategoricalVariable(variables=cfg.CATEGORICAL_VARIABLES_ENCODE)),
    ("transformer_logarithm",
     ps.TransformerLogarithm(variables=cfg.NUMERICAL_VARIABLES_LOGARITHM_TRANSFORM)),
    ("drop_selected_variable",
     ps.DropSelectedVariable(variables=cfg.DROP_VARIABLES)),
    ("scaler_minmax",
     MinMaxScaler()),
    ("linear_regression",
     Lasso(alpha=0.005, random_state=0))
])
