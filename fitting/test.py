import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Sample data (replace this with your actual data)
data = {
    'subject': [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6],
    'stage': ['Stage1']*6 + ['Stage2']*6 + ['Stage3']*6,
    'model': ['Model1', 'Model2', 'Model3', 'Model1', 'Model2', 'Model3']*3,
    'likelihood': [0.8, 0.7, 0.9, 0.6, 0.5, 0.7, 0.7, 0.6, 0.8, 0.9, 0.8, 0.7, 0.5, 0.4, 0.6, 0.9, 0.8, 0.7]
}


df = pd.DataFrame(data)

# Fit the mixed-effects model
model_formula = "likelihood ~ model * stage"
mixedlm_model = smf.mixedlm(model_formula, df, groups=df['subject'])
mixedlm_result = mixedlm_model.fit()

# Print the summary
print(mixedlm_result.summary())


