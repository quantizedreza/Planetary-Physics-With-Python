import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm



df = pd.read_csv("data.csv")
annual = df.groupby("Year")["Temperature"].mean().reset_index(name="T_avg")

years = annual["Year"].astype(float).values
y_obs = annual["T_avg"].values
t     = years - years[0]

mean_T = y_obs.mean()
slope_ols = np.polyfit(t, y_obs, 1)[0]
resid = y_obs - (mean_T + slope_ols * t)
sigma = np.std(resid)

np.random.seed(2025)
N_SIM = 400
sims = mean_T + np.random.normal(0, sigma, size=(N_SIM, len(years)))
lower = np.percentile(sims,  2.5, axis=0)
upper = np.percentile(sims, 97.5, axis=0)


with plt.xkcd(scale=1.4, length=220, randomness=2.2):
    plt.figure(figsize=(11, 6.5))

    plt.fill_between(years, lower, upper,
                     color='lightblue', alpha=0.22, label=f'noise cloud (±{sigma:.2f} °F)')

    for i in np.random.choice(N_SIM, 4, replace=False):
        plt.plot(years, sims[i], color='steelblue', lw=1.3, alpha=0.5)

    plt.plot(years, y_obs, 'o-', color='darkred', markersize=8, lw=1.8,
             label=f'real data (slope {slope_ols:+.4f})')

    x_edge = np.array([years.min(), years.max()])
    plt.plot(x_edge, mean_T + 0.05 * (x_edge - years[0]), '--', color='goldenrod', lw=3,
             label='+0.05 °F/yr (tiny!)')

    plt.axhline(mean_T, color='0.4', ls='--', lw=1.4)

    plt.title("is +0.05 °F/year even detectable?")
    plt.xlabel("year")
    plt.ylabel("temperature (°F)")

    plt.legend(loc='upper left', fontsize=11)
    plt.grid(alpha=0.2)

    plt.annotate("Look\nthere is a trend!",
                 xy=(years.mean(), mean_T + sigma),
                 xytext=(years.min()+10, mean_T + 2.2*sigma),
                 arrowprops=dict(arrowstyle='->'), fontsize=12)

    plt.tight_layout()
    plt.show()
