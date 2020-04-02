import wrangle
import split_scale
import matplotlib.pyplot as plt
import seaborn as sns

def plot_variable_pairs(df):
    sns.pairplot(df, kind='reg', plot_kws={'line_kws':{'color':'orange'}, 'scatter_kws': {'alpha': 0.1}})
    plt.show()

def months_to_years(months_column, df):
    df['tenure_years'] = (months_column / 12).astype(int)
    return df

def plot_categorical_continuous_vars(categorical_var, continuous_var, df):
    f, axes = plt.subplots(3, 1, figsize=(16, 16))

    sns.boxplot(x=categorical_var, y=continuous_var, data=df, ax=axes[0])
    sns.swarmplot(x=categorical_var, y=continuous_var, data=df, color='.2', alpha=.7, ax=axes[0])
    sns.violinplot(x=categorical_var, y=continuous_var, data=df, inner='stick', ax=axes[1])
    sns.barplot(x=categorical_var, y=continuous_var, data=df, capsize=.2, ax=axes[2])
    plt.show()