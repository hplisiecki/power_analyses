import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from tqdm import tqdm

# Monte Carlo power simulation for H1–H4 in Study 1

# 1. Parameters
sigma_u = 1.5            # variability of participants’ random intercepts
sigma_e = 2.0            # residual (within-cell) noise
beta0   = 5.0            # overall intercept
beta_content = -0.2      # true “small” content effect for H1
beta_h2      = -0.2      # extra drop for male→male (H2 and H3)
beta_h4      = +0.2      # boost for male→female (H4)
alpha = 0.05             # Type I error rate


# 2. Simulation settings
Ns = np.arange(1000, 3001, 200)  # sample sizes to sweep
B = 100  # Monte Carlo replicates per N


# 3. Helper: simulate one dataset of size N
def simulate_dataset(N):
    # build design grid
    rows = []
    for pid in range(N):
        for content in ['neutral', 'stereotypical']:
            for author in ['female', 'male']:
                for partner in ['neutral', 'female', 'male']:
                    rows.append((pid, content, author, partner))
    df = pd.DataFrame(rows, columns=['participant', 'content', 'author', 'partner'])

    # random intercepts
    u = np.random.normal(0, sigma_u, size=N)
    df['u'] = df['participant'].map(dict(enumerate(u)))

    # fixed effects
    fe = beta0
    fe_arr = np.full(len(df), beta0)
    fe_arr += (df['content'] == 'stereotypical') * beta_content
    fe_arr += ((df['content'] == 'stereotypical') &
               (df['author'] == 'male') &
               (df['partner'] == 'male')) * beta_h2
    fe_arr += ((df['content'] == 'stereotypical') &
               (df['author'] == 'male') &
               (df['partner'] == 'female')) * beta_h4

    # simulate outcome
    eps = np.random.normal(0, sigma_e, size=len(df))
    df['amount'] = fe_arr + df['u'] + eps

    # cast factors
    df['content'] = df['content'].astype('category')
    df['author'] = df['author'].astype('category')
    df['partner'] = df['partner'].astype('category')

    return df


# 4. Run Monte Carlo
results = []
for N in Ns:
    print(f"Running simulations for N={N}...")
    rejects = {'H1': 0, 'H2': 0, 'H3': 0, 'H4': 0}
    for _ in tqdm(range(B)):
        df = simulate_dataset(N)
        model = smf.mixedlm("amount ~ content * author * partner",
                            groups="participant", data=df)
        fit = model.fit(reml=False)
        p = fit.pvalues

        # H1
        if p.get('content[T.stereotypical]', 1) < alpha:
            rejects['H1'] += 1
        # H2 & H3: same 3-way term
        key_h2 = 'content[T.stereotypical]:author[T.male]:partner[T.male]'
        if p.get(key_h2, 1) < alpha:
            rejects['H2'] += 1
            rejects['H3'] += 1
        # H4
        key_h4 = 'content[T.stereotypical]:author[T.male]'
        if p.get(key_h4, 1) < alpha:
            rejects['H4'] += 1

    # collect power estimates
    results.append({
        'N': N,
        'Power_H1': rejects['H1'] / B,
        'Power_H2': rejects['H2'] / B,
        'Power_H3': rejects['H3'] / B,
        'Power_H4': rejects['H4'] / B,
    })

df_power = pd.DataFrame(results)

# 5. Plot
plt.figure(figsize=(8, 5))
for h in ['H1', 'H2', 'H3', 'H4']:
    plt.plot(df_power['N'], df_power[f'Power_{h}'], marker='o', label=h)
plt.xlabel("Number of Participants")
plt.ylabel("Estimated Power")
plt.title("Monte Carlo Power Curves for H1–H4")
plt.legend()
plt.grid(True)
plt.ylim(0, 1)
plt.show()

# save plot
plt.savefig("power_curves_study1.png", dpi=300, bbox_inches='tight')

# save
df_power.to_csv("power_estimates_study1.csv", index=False)

# df_power contains the numeric power by N for each hypothesis
