"""Train and compare four regression models with 5-fold cross-validation.

Models: Linear Regression, Decision Tree, Random Forest, XGBoost.
Target: log(Price). Metrics (R2, MAE, RMSE, MAPE) computed on peso scale.
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

try:
    from xgboost import XGBRegressor
    XGB_AVAILABLE = True
except ImportError:
    from sklearn.ensemble import GradientBoostingRegressor
    XGB_AVAILABLE = False
    print("XGBoost not installed, using sklearn GradientBoostingRegressor")

# ---- Data prep ----
df = pd.read_csv('PH_Housing_Cleaned.csv')

Q1, Q3 = df['Price'].quantile(0.25), df['Price'].quantile(0.75)
IQR = Q3 - Q1
df = df[(df['Price'] >= Q1 - 1.5 * IQR) & (df['Price'] <= Q3 + 1.5 * IQR)].copy()
for col in ['Floor Area', 'Land Area']:
    df[col] = df[col].clip(upper=df[col].quantile(0.99))
df['Log_Price'] = np.log(df['Price'])
print(f"Regression dataset: {len(df)} records")

features = ['Bedrooms', 'Bathrooms', 'Floor Area', 'Land Area', 'Latitude', 'Longitude']
X = df[features].values
y_log = df['Log_Price'].values
y_actual = df['Price'].values

# ---- Model definitions ----
if XGB_AVAILABLE:
    xgb_model = XGBRegressor(
        n_estimators=300, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
        reg_alpha=0.01, reg_lambda=1.0, random_state=42, verbosity=0
    )
else:
    xgb_model = GradientBoostingRegressor(
        n_estimators=300, max_depth=6, learning_rate=0.1,
        subsample=0.8, min_samples_leaf=3, random_state=42
    )

models = {
    'Multiple Linear\nRegression': {
        'model': LinearRegression(),
        'needs_scaling': True,
        'color': '#1E88E5',
    },
    'Decision\nTree': {
        'model': DecisionTreeRegressor(max_depth=10, random_state=42),
        'needs_scaling': False,
        'color': '#43A047',
    },
    'Random\nForest': {
        'model': RandomForestRegressor(
            n_estimators=200, max_depth=15, min_samples_split=5,
            min_samples_leaf=2, max_features=0.8, random_state=42, n_jobs=-1),
        'needs_scaling': False,
        'color': '#FF9800',
    },
    'XGBoost': {
        'model': xgb_model,
        'needs_scaling': False,
        'color': '#E53935',
    },
}

# ---- 5-Fold Cross-Validation ----
kf = KFold(n_splits=5, shuffle=True, random_state=42)
all_results = {}

for name, cfg in models.items():
    fold_data = {'r2': [], 'mae': [], 'rmse': [], 'mape': [],
                 'actual': [], 'predicted': []}
    importances_sum = np.zeros(len(features))
    has_importance = False

    for train_idx, test_idx in kf.split(X):
        Xtr, Xte = X[train_idx], X[test_idx]
        if cfg['needs_scaling']:
            sc = StandardScaler()
            Xtr = sc.fit_transform(Xtr)
            Xte = sc.transform(Xte)

        m = type(cfg['model'])(**cfg['model'].get_params())
        m.fit(Xtr, y_log[train_idx])

        pred_log = m.predict(Xte)
        # Clip log predictions to prevent exp() from blowing up on LR extrapolations
        pred_log = np.clip(pred_log, y_log.min() - 0.5, y_log.max() + 0.5)
        pred_pesos = np.exp(pred_log)
        actual = y_actual[test_idx]

        fold_data['r2'].append(r2_score(actual, pred_pesos))
        fold_data['mae'].append(mean_absolute_error(actual, pred_pesos))
        fold_data['rmse'].append(np.sqrt(mean_squared_error(actual, pred_pesos)))
        fold_data['mape'].append(np.mean(np.abs((actual - pred_pesos) / actual)) * 100)
        fold_data['actual'].extend(actual.tolist())
        fold_data['predicted'].extend(pred_pesos.tolist())

        if hasattr(m, 'feature_importances_'):
            importances_sum += m.feature_importances_
            has_importance = True

    fold_data['importance'] = dict(zip(features, (importances_sum / 5).tolist())) if has_importance else None
    all_results[name] = fold_data

    short = name.replace('\n', ' ')
    print(f"{short:<25s} R2={np.mean(fold_data['r2']):.4f}+/-{np.std(fold_data['r2']):.3f}  "
          f"MAE=P{np.mean(fold_data['mae'])/1e6:.2f}M  "
          f"MAPE={np.mean(fold_data['mape']):.2f}%")

# ---- Figure 1: Metric comparison bars ----
names = list(all_results.keys())
colors = [models[n]['color'] for n in names]

fig, axes = plt.subplots(1, 4, figsize=(20, 5))
metrics_plot = [
    ('r2', 'R-squared', 'higher is better', 1),
    ('mae', 'MAE (Millions P)', 'lower is better', 1e6),
    ('rmse', 'RMSE (Millions P)', 'lower is better', 1e6),
    ('mape', 'MAPE (%)', 'lower is better', 1),
]
for ax, (key, label, direction, divisor) in zip(axes, metrics_plot):
    vals = [np.mean(all_results[n][key]) / divisor for n in names]
    errs = [np.std(all_results[n][key]) / divisor for n in names]
    bars = ax.bar(names, vals, yerr=errs, color=colors, edgecolor='white', capsize=5)
    ax.set_ylabel(label)
    ax.set_title(f'{label}\n({direction})', fontweight='bold')
    for bar, v in zip(bars, vals):
        fmt = f'{v:.4f}' if key == 'r2' else (f'{v:.1f}%' if key == 'mape' else f'P{v:.1f}M')
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.02,
                fmt, ha='center', fontsize=8, fontweight='bold')
    ax.tick_params(axis='x', labelsize=8)

plt.suptitle('Model Performance Comparison (5-Fold CV)', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('fig_model_comparison_bar.png', dpi=200, bbox_inches='tight')
plt.close()

# ---- Figure 2: Predicted vs Actual ----
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.flatten()
for ax, (name, data) in zip(axes, all_results.items()):
    short = name.replace('\n', ' ')
    actual = np.array(data['actual'])
    predicted = np.array(data['predicted'])
    color = models[name]['color']
    r2 = np.mean(data['r2'])
    mape = np.mean(data['mape'])

    ax.scatter(actual / 1e6, predicted / 1e6, alpha=0.3, s=12, color=color)
    mx = max(actual.max(), predicted.max()) / 1e6
    ax.plot([0, mx], [0, mx], 'r--', linewidth=1.5, label='Perfect Prediction')
    ax.set_xlabel('Actual Price (Millions P)')
    ax.set_ylabel('Predicted Price (Millions P)')
    ax.set_title(f'{short}\nR2={r2:.4f} | MAPE={mape:.1f}%', fontweight='bold')
    ax.legend()
    ax.set_xlim(0, min(mx * 1.05, 60))
    ax.set_ylim(0, min(mx * 1.05, 60))

plt.suptitle('Predicted vs. Actual Price', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('fig_pred_vs_actual_all.png', dpi=200, bbox_inches='tight')
plt.close()

# ---- Figure 3: CV boxplot ----
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
metric_keys = [('r2', 'R-squared', 1), ('mae', 'MAE (M P)', 1e6),
               ('rmse', 'RMSE (M P)', 1e6), ('mape', 'MAPE (%)', 1)]
for ax, (key, label, divisor) in zip(axes, metric_keys):
    data_list = [[v / divisor for v in all_results[n][key]] for n in names]
    bp = ax.boxplot(data_list, tick_labels=names, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel(label)
    ax.set_title(label, fontweight='bold')
    ax.tick_params(axis='x', labelsize=8)

plt.suptitle('5-Fold CV Score Distribution', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('fig_cv_boxplot.png', dpi=200, bbox_inches='tight')
plt.close()

# ---- Figure 4: Feature importance (tree-based only) ----
fi_models = {n.replace('\n', ' '): d['importance']
             for n, d in all_results.items() if d['importance'] is not None}
n_fi = len(fi_models)
fig, axes = plt.subplots(1, n_fi, figsize=(6 * n_fi, 5))
if n_fi == 1: axes = [axes]
fi_colors = ['#43A047', '#FF9800', '#E53935']

for ax, (mname, fi), color in zip(axes, fi_models.items(), fi_colors):
    imp = pd.Series(fi).sort_values(ascending=True)
    imp.plot(kind='barh', color=color, ax=ax, edgecolor='white')
    ax.set_xlabel('Importance')
    ax.set_title(mname, fontweight='bold')

plt.suptitle('Feature Importance Comparison', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('fig_feature_importance_comparison.png', dpi=200, bbox_inches='tight')
plt.close()

# ---- Figure 5: Residuals ----
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()
for ax, (name, data) in zip(axes, all_results.items()):
    short = name.replace('\n', ' ')
    actual = np.array(data['actual'])
    predicted = np.array(data['predicted'])
    residuals = (actual - predicted) / 1e6
    color = models[name]['color']

    ax.hist(residuals, bins=40, color=color, edgecolor='white', alpha=0.85)
    ax.axvline(0, color='red', linestyle='--', linewidth=1.5)
    ax.set_xlabel('Residual (Millions P)')
    ax.set_ylabel('Frequency')
    ax.set_title(f'{short} Residuals', fontweight='bold')

plt.suptitle('Residual Distributions', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('fig_residuals_all.png', dpi=200, bbox_inches='tight')
plt.close()

# ---- Save JSON summary ----
summary = {}
for name, data in all_results.items():
    short = name.replace('\n', ' ')
    summary[short] = {
        'r2': f"{np.mean(data['r2']):.4f} +/- {np.std(data['r2']):.4f}",
        'mae': f"P{np.mean(data['mae'])/1e6:.2f}M +/- P{np.std(data['mae'])/1e6:.2f}M",
        'rmse': f"P{np.mean(data['rmse'])/1e6:.2f}M +/- P{np.std(data['rmse'])/1e6:.2f}M",
        'mape': f"{np.mean(data['mape']):.2f}% +/- {np.std(data['mape']):.2f}%",
    }
with open('results_regression_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print("\nAll figures and results_regression_summary.json saved.")
