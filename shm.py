import pickle
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# ============================================================
# Configuration & Helpers
# ============================================================
PICKLE_PATH = Path("/home/linyuan/Desktop/yk_project/month/measurements 2018_06.pickle")
OUT_DIR     = Path("/home/linyuan/Desktop/yk_project/final_report/")
EDA_DIR, ANALYSIS_DIR, TABLE_DIR = OUT_DIR/"eda", OUT_DIR/"analysis", OUT_DIR/"tables"
for d in (EDA_DIR, ANALYSIS_DIR, TABLE_DIR): d.mkdir(parents=True, exist_ok=True)

SENSOR_ID, SENSOR_NAME = 0, "Path 5-1"
ENV_VARS = ["temperature", "pressure", "brightness", "humidity"]
ENV_LABELS = {"temperature": "Temperature (°C)", "pressure": "Pressure (hPa)", 
              "brightness": "Brightness", "humidity": "Humidity (%)"}
MAX_SCATTER = 5000
np.random.seed(42)

def save_plot(filepath):
    """Helper to avoid repeating plt.tight_layout() and savefig boilerplate."""
    plt.tight_layout()
    plt.savefig(filepath, dpi=130)
    plt.close()

# ============================================================
# Main Analysis Steps
# ============================================================
def load_data():
    print("=== Step 1: Loading data ===")
    with open(PICKLE_PATH, "rb") as f: d = pickle.load(f)
    
    gw_one = d["guided wave"][:, SENSOR_ID, :].astype(np.float64)
    times = pd.to_datetime(d["datatime"])
    rms = np.sqrt((gw_one ** 2).mean(axis=1))
    tod = pd.cut(times.hour, bins=[-0.1, 6, 12, 18, 24], labels=["night", "morning", "afternoon", "evening"])

    df = pd.DataFrame({**{v: d[v] for v in ENV_VARS}, "time": times, "RMS": rms, 
                       "log_RMS": np.log(rms), "time_of_day": tod})
    
    print(f"  Sensor: {SENSOR_ID} ({SENSOR_NAME})\n  Number of measurements: {len(rms)}")
    print(f"  RMS range: [{rms.min():.6f}, {rms.max():.6f}]\n  Time range: {times.min()} -> {times.max()}")
    print(f"  Samples by time_of_day:\n{df['time_of_day'].value_counts().sort_index().to_string()}")
    return df

def eda(df):
    print("\n=== Step 2: EDA ===")
    # 2.1 Env Summary
    env_summary = df[ENV_VARS].describe().round(3)
    env_summary.to_csv(TABLE_DIR / "eda_env_summary.csv")
    
    # 2.2 Env Histograms
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, var in zip(axes.flatten(), ENV_VARS):
        x = df[var].values
        ax.hist(x, bins=50, color="steelblue", edgecolor="white")
        ax.axvline(x.mean(), color="red", linestyle="--", label=f"mean = {x.mean():.2f}")
        ax.set(xlabel=ENV_LABELS[var], ylabel="Frequency", title=f"{var}: mean={x.mean():.2f}, SD={x.std():.2f}")
        ax.legend()
    fig.suptitle("Distributions of Environmental Variables", fontsize=14)
    save_plot(EDA_DIR / "fig01_env_histograms.png")

    # 2.3 RMS Distribution
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for i, col in enumerate(["RMS", "log_RMS"]):
        axes[i].hist(df[col], bins=80, color=["darkorange", "seagreen"][i], edgecolor="white")
        axes[i].set(xlabel=col, ylabel="Frequency", title=f"{col} — skewness = {stats.skew(df[col]):.2f}")
    save_plot(EDA_DIR / "fig02_rms_distribution.png")

    pd.DataFrame({"statistic": ["n", "mean", "std", "min", "max", "skewness"],
                  "log_RMS": [len(df), df["log_RMS"].mean(), df["log_RMS"].std(), df["log_RMS"].min(), 
                              df["log_RMS"].max(), stats.skew(df["log_RMS"])]
                 }).round(4).to_csv(TABLE_DIR / "eda_logrms_summary.csv", index=False)

    # 2.4 Scatterplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    df_plot = df.sample(n=min(MAX_SCATTER, len(df)), replace=False)
    corr_rows = []
    for ax, var in zip(axes.flatten(), ENV_VARS):
        ax.scatter(df_plot[var], df_plot["log_RMS"], s=4, alpha=0.4, color="navy")
        r, p = stats.pearsonr(df[var], df["log_RMS"])
        ax.set(xlabel=ENV_LABELS[var], ylabel="log(RMS)", title=f"{var}: Pearson r = {r:+.3f}, {'p < 0.001' if p < 0.001 else f'p = {p:.3f}'}")
        ax.grid(alpha=0.3)
        corr_rows.append({"variable": var, "pearson_r": r, "p_value": p})
    fig.suptitle(f"log(RMS) vs Environmental Variables (Sensor {SENSOR_ID}, {SENSOR_NAME})", fontsize=14)
    save_plot(EDA_DIR / "fig03_logrms_vs_env_scatter.png")
    pd.DataFrame(corr_rows).round(4).to_csv(TABLE_DIR / "eda_pearson_correlations.csv", index=False)

    # 2.5 Time of Day
    tod_counts = df["time_of_day"].value_counts().sort_index()
    tod_pct = (tod_counts / len(df) * 100).round(2)
    pd.DataFrame({"count": tod_counts, "percent": tod_pct}).to_csv(TABLE_DIR / "eda_time_of_day_proportions.csv")
    
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(tod_counts.index.astype(str), tod_counts.values, color=["#3b528b", "#21918c", "#5ec962", "#fde725"], edgecolor="black")
    for bar, c, p in zip(bars, tod_counts.values, tod_pct.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f"{c}\n({p:.1f}%)", ha="center", va="bottom", fontsize=10)
    ax.set(xlabel="Time of day", ylabel="Number of obs", title=f"Distribution by Time of Day (N = {len(df)})", ylim=(0, tod_counts.max()*1.15))
    ax.grid(alpha=0.3, axis="y")
    save_plot(EDA_DIR / "fig04a_time_of_day_proportions.png")

    fig, ax = plt.subplots(figsize=(9, 5))
    df.boxplot(column="log_RMS", by="time_of_day", ax=ax, grid=True, showfliers=False)
    ax.set(xlabel="Time of day", ylabel="log(RMS)", title=f"log(RMS) by Time of Day (Sensor {SENSOR_ID}, {SENSOR_NAME})")
    plt.suptitle("")
    tod_summary = df.groupby("time_of_day", observed=True)["log_RMS"].agg(["count", "mean", "std"]).round(4)
    tod_summary.to_csv(TABLE_DIR / "eda_logrms_by_time_of_day.csv")
    save_plot(EDA_DIR / "fig04_logrms_by_time_of_day_boxplot.png")

    df[ENV_VARS].corr().round(3).to_csv(TABLE_DIR / "eda_env_corr_matrix.csv")

def regression_analysis(df):
    print("\n=== Step 3: Regression ===")
    X = sm.add_constant(df[ENV_VARS].astype(float))
    model = sm.OLS(df["log_RMS"], X).fit()
    
    ci = model.conf_int().rename(columns={0: "CI_2.5%", 1: "CI_97.5%"})
    coef_table = pd.DataFrame({"estimate": model.params, "std_err": model.bse, 
                               "t_stat": model.tvalues, "p_value": model.pvalues}).join(ci).round(6)
    coef_table.to_csv(TABLE_DIR / "regression_coefficients.csv")
    (TABLE_DIR / "regression_summary.txt").write_text(str(model.summary()))

    # Assumptions
    fig, ax = plt.subplots(figsize=(6, 6))
    sm.qqplot(model.resid, line="45", fit=True, ax=ax)
    ax.set_title("QQ Plot of Residuals (Normality)")
    save_plot(ANALYSIS_DIR / "fig06_qqplot_residuals.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    idx = np.random.choice(len(model.resid), min(MAX_SCATTER, len(model.resid)), replace=False)
    ax.scatter(model.fittedvalues[idx], model.resid[idx], s=4, alpha=0.3, color="navy")
    ax.axhline(0, color="red", linestyle="--")
    ax.set(xlabel="Fitted values", ylabel="Residuals", title="Residuals vs Fitted (Homoscedasticity)")
    ax.grid(alpha=0.3)
    save_plot(ANALYSIS_DIR / "fig07_residuals_vs_fitted.png")

    X_vif = X.values
    vif = pd.DataFrame({"variable": X.columns, "VIF": [variance_inflation_factor(X_vif, i) for i in range(X_vif.shape[1])]}).round(3)
    vif.to_csv(TABLE_DIR / "regression_vif.csv", index=False)

    (TABLE_DIR / "regression_assumptions.txt").write_text(
        f"=== Assumption Checks ===\nSample size: {len(df)}\n1) Normality: QQ plot\n2) Homoscedasticity: Res vs Fit\n3) Multicollinearity (VIF)\n{vif.to_string(index=False)}\nRule of thumb: VIF > 5 => concern.\n"
    )
    return model

def anova_analysis(df):
    print("\n=== Step 4: ANOVA ===")
    levels = list(df["time_of_day"].cat.categories)
    groups = [df.loc[df["time_of_day"] == lv, "log_RMS"].values for lv in levels]
    
    f_stat, p_val = stats.f_oneway(*groups)
    grp_summary = df.groupby("time_of_day", observed=True)["log_RMS"].agg(["count", "mean", "std"]).round(4)
    tukey = pairwise_tukeyhsd(df["log_RMS"], df["time_of_day"])
    
    (TABLE_DIR / "anova_tukey.txt").write_text(
        f"=== ANOVA ===\nF = {f_stat:.4f}, p = {p_val:.4e}\n\nGroup stats:\n{grp_summary}\n\nTukey HSD:\n{tukey}"
    )

    lev_stat, lev_p = stats.levene(*groups)
    fig, axes = plt.subplots(1, len(levels), figsize=(4 * len(levels), 4))
    for ax, lv, g in zip(axes, levels, groups):
        sm.qqplot(np.random.choice(g, min(2000, len(g)), replace=False), line="45", fit=True, ax=ax)
        ax.set_title(f"{lv} (n={len(g)})")
    fig.suptitle("QQ Plots by Time of Day", fontsize=14)
    save_plot(ANALYSIS_DIR / "fig08_qqplots_by_time_of_day.png")

    (TABLE_DIR / "anova_assumptions.txt").write_text(
        f"=== ANOVA Assumptions ===\n1) Equal variance (Levene): stat={lev_stat:.4f}, p={lev_p:.4e}\n2) Normality: per-group QQ plots.\n"
    )

def write_final_summary(df, model):
    print("\n=== Step 5: FINAL_SUMMARY ===")
    lines = [
        "=" * 70, "FINAL REPORT - Statistical Analysis Summary", "=" * 70,
        f"\nData: {PICKLE_PATH.name} | Sensor: {SENSOR_ID} ({SENSOR_NAME}) | N = {len(df)} | Time: {df['time'].min()} -> {df['time'].max()}",
        f"\nY = log(RMS): mean = {df['log_RMS'].mean():.4f}, SD = {df['log_RMS'].std():.4f}",
        f"\nRegression: R²={model.rsquared:.4f}, adj R²={model.rsquared_adj:.4f}, F={model.fvalue:.2f}, p={model.f_pvalue:.2e}\nCoefficients:"
    ]
    for v in ENV_VARS:
        est, p = model.params[v], model.pvalues[v]
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        lines.append(f"  {v:12s}: {est:+.6f}  {'↑' if est > 0 else '↓'}  p={p:.2e}  [{sig}]")
    
    (OUT_DIR / "FINAL_SUMMARY.txt").write_text("\n".join(lines))

def collect_report_assets():
    print("\n=== Step 6: Collecting assets ===")
    REPORT_DIR = OUT_DIR / "report_assets"
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    assets = [
        "eda/fig01_env_histograms.png", "eda/fig02_rms_distribution.png", "eda/fig03_logrms_vs_env_scatter.png",
        "eda/fig04a_time_of_day_proportions.png", "eda/fig04_logrms_by_time_of_day_boxplot.png",
        "analysis/fig06_qqplot_residuals.png", "analysis/fig07_residuals_vs_fitted.png", "analysis/fig08_qqplots_by_time_of_day.png",
        "tables/eda_env_summary.csv", "tables/eda_logrms_by_time_of_day.csv", "tables/eda_pearson_correlations.csv",
        "tables/regression_coefficients.csv", "tables/anova_tukey.txt", "FINAL_SUMMARY.txt",
        "tables/regression_assumptions.txt", "tables/anova_assumptions.txt"
    ]
    
    for asset in assets:
        src = OUT_DIR / asset
        if src.exists(): shutil.copy(src, REPORT_DIR / src.name)

    (REPORT_DIR / "INDEX.txt").write_text("REPORT ASSETS: Contains 8 figures + 5 tables + 3 text files for the final report.")

if __name__ == "__main__":
    df = load_data()
    model = regression_analysis(df)
    anova_analysis(df)
    write_final_summary(df, model)
    collect_report_assets()
    print(f"\nAll done! Results in: {OUT_DIR}")