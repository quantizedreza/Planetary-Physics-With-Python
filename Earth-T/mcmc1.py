import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# ────────────────────────────────────────────────
#  SETTINGS & HYPERPARAMETERS
# ────────────────────────────────────────────────
RANDOM_SEED       = 2025
N_ITER            = 50_000
BURN_IN           = 10_000
THIN              = 5
PROPOSAL_SCALE    = {'intercept': 1.2, 'slope': 0.015, 'sigma': 0.35}

np.random.seed(RANDOM_SEED)

# ────────────────────────────────────────────────
#  1. Load data & compute national annual means
# ────────────────────────────────────────────────
df = pd.read_csv("data.csv")

# Create a proper date and sort
df["Date"] = pd.to_datetime(df[["Year", "Month"]].assign(day=1))
df = df.sort_values(["State", "Date"])

# Annual national average
annual = df.groupby("Year")["Temperature"].mean().reset_index()
annual = annual.rename(columns={"Temperature": "T_avg"})

years = annual["Year"].values
y     = annual["T_avg"].values

# Center years for numerical stability
t = years - years[0]               # years since first year
n = len(y)

print(f"Years: {years.min()} – {years.max()}    (n = {n})")
print(f"Mean temperature: {y.mean():.2f} °F\n")

# ────────────────────────────────────────────────
#  2. Log posterior (up to constant)
# ────────────────────────────────────────────────
def log_posterior(intercept, slope, log_sigma, t, y):
    sigma = np.exp(log_sigma)
    if sigma <= 0:
        return -np.inf

    mu = intercept + slope * t
    log_lik = stats.norm.logpdf(y, loc=mu, scale=sigma).sum()

    # Priors (weakly informative)
    log_prior_a     = stats.norm.logpdf(intercept,  52, 10)
    log_prior_b     = stats.norm.logpdf(slope,     0.04, 0.08)
    log_prior_logsig = stats.norm.logpdf(log_sigma, np.log(1.2), 1.0)

    return log_lik + log_prior_a + log_prior_b + log_prior_logsig


# ────────────────────────────────────────────────
#  3. Metropolis-Hastings sampler
# ────────────────────────────────────────────────
def metropolis_hastings():
    # Initial guess (rough OLS)
    slope_init    = np.polyfit(t, y, 1)[0]
    intercept_init = y.mean() - slope_init * t.mean()
    resid = y - (intercept_init + slope_init * t)
    sigma_init = np.std(resid)
    log_sigma_init = np.log(sigma_init)

    current = np.array([intercept_init, slope_init, log_sigma_init])
    samples = []

    n_accepted = 0

    for i in range(N_ITER):
        proposal = current + np.random.normal(0, [
            PROPOSAL_SCALE['intercept'],
            PROPOSAL_SCALE['slope'],
            PROPOSAL_SCALE['sigma']
        ])

        log_p_curr = log_posterior(*current, t, y)
        log_p_prop = log_posterior(*proposal, t, y)

        log_alpha = log_p_prop - log_p_curr

        if np.log(np.random.rand()) < log_alpha:
            current = proposal
            n_accepted += 1

        if i >= BURN_IN and (i - BURN_IN) % THIN == 0:
            samples.append(current.copy())

        if (i + 1) % 5000 == 0:
            acc_rate = n_accepted / (i + 1)
            print(f"Iter {i+1:6d} | acc = {acc_rate:.1%} | "
                  f"b = {current[1]:+.4f} | σ = {np.exp(current[2]):.2f}")

    samples = np.array(samples)
    print(f"\nFinal acceptance rate: {n_accepted / N_ITER:.1%}")
    return samples


# ────────────────────────────────────────────────
#  4. Run MCMC
# ────────────────────────────────────────────────
print("Running MCMC ...\n")
posterior_samples = metropolis_hastings()

intercept_samples = posterior_samples[:, 0]
slope_samples     = posterior_samples[:, 1]
sigma_samples     = np.exp(posterior_samples[:, 2])

# ────────────────────────────────────────────────
#  5. Summarize results
# ────────────────────────────────────────────────
print("\n" + "═"*65)
print("POSTERIOR SUMMARY  (after burn-in & thinning)")
print("═"*65)
print(f"Intercept     {np.mean(intercept_samples):8.2f} ± {np.std(intercept_samples):.2f}")
print(f"Slope (°F/yr) {np.mean(slope_samples):8.4f} ± {np.std(slope_samples):.4f}")
print(f"Sigma         {np.mean(sigma_samples):8.2f} ± {np.std(sigma_samples):.2f}")
print()

p_warming = (slope_samples > 0).mean()
ci95 = np.percentile(slope_samples, [2.5, 97.5])
print(f"P(slope > 0)  = {p_warming:.4f}")
print(f"95% CrI slope = [{ci95[0]:+.4f}, {ci95[1]:+.4f}] °F/year")
print("═"*65 + "\n")

# ────────────────────────────────────────────────
#  6. Visualisation
# ────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), height_ratios=[3,1])

# Trace plot – slope
ax2.plot(slope_samples, lw=0.7, color='C0', alpha=0.8)
ax2.set_ylabel("slope (°F/yr)")
ax2.set_xlabel("MCMC sample index (post burn-in)")
ax2.grid(True, alpha=0.3)

# Data + posterior predictive
ax1.scatter(years, y, c='C0', s=60, alpha=0.9, label="Annual national mean")

# Plot 100 random posterior lines
for i in np.random.choice(len(slope_samples), 120, replace=False):
    a, b = intercept_samples[i], slope_samples[i]
    ax1.plot(years, a + b * t, color='salmon', alpha=0.07, lw=1.1, zorder=1)

# Mean trend
a_mean = np.mean(intercept_samples)
b_mean = np.mean(slope_samples)
ax1.plot(years, a_mean + b_mean * t, 'k', lw=2.4, label=f"Posterior mean trend\n({b_mean:+.4f} °F/yr)")

ax1.set_title("U.S. National Annual Mean Temperature + Bayesian Trend (MCMC)", fontsize=14, pad=12)
ax1.set_ylabel("Temperature (°F)")
ax1.legend(loc="upper left", frameon=True)
ax1.grid(True, alpha=0.35)

plt.tight_layout()
plt.show()
