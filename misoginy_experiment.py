import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.formula.api import mixedlm
from scipy import stats
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

# Ensure a spawn start method for compatibility
multiprocessing.set_start_method('spawn', force=True)


# --- Simulation & Testing Functions (pickleable at module level) ---

def simulate_dataset(n_per_cond, tau, sigma, d, seed):
    np.random.seed(seed)
    conds = ['SM', 'SF', 'NM', 'NF']
    partners = ['male', 'female', 'neutral']
    N = n_per_cond * len(conds)

    subjects = pd.DataFrame({
        'subj': np.arange(N),
        'cond': np.repeat(conds, n_per_cond)
    })
    subjects['b0'] = np.random.normal(0, tau, size=N)

    df = subjects.merge(pd.DataFrame({'partner': partners}), how='cross')
    fe = np.zeros(len(df))

    # H1: stereotype comment ↓ neutral-partner trust
    fe[(df.partner == 'neutral') & df.cond.isin(['SM', 'SF'])] -= d
    # H2: SM comment ↓ male-partner trust
    fe[(df.partner == 'male') & (df.cond == 'SM')] -= d
    # H3: SF comment on female-partner ↑ (relative coding)
    fe[(df.partner == 'female') & (df.cond == 'SF')] += d
    # H4: SM comment ↑ female-partner trust
    fe[(df.partner == 'female') & (df.cond == 'SM')] += d

    df['y'] = df.b0 + fe + np.random.normal(0, sigma, size=len(df))
    return df


def fit_and_test(df):
    md = mixedlm("y ~ C(cond)*C(partner)", df, groups=df["subj"])
    mdf = md.fit(reml=False, disp=False)
    params = mdf.params
    cov = mdf.cov_params()
    cols = params.index.tolist()

    def contrast(vec):
        est = vec.dot(params)
        se = np.sqrt(vec @ cov @ vec)
        z = est / se
        return 2 * (1 - stats.norm.cdf(abs(z)))

    def make_vec(cond, partner, w=1.0):
        v = np.zeros(len(cols))
        v[cols.index('Intercept')] = 1
        c_name = f"C(cond)[T.{cond}]"
        p_name = f"C(partner)[T.{partner}]"
        i_name = f"{c_name}:{p_name}"
        for name in (c_name, p_name, i_name):
            if name in cols:
                v[cols.index(name)] = 1
        return w * v

    c1 = (make_vec('SM', 'neutral', .5) + make_vec('SF', 'neutral', .5)
          - make_vec('NM', 'neutral', .5) - make_vec('NF', 'neutral', .5))
    c2 = (make_vec('SM', 'male', 1) +
          sum(make_vec(c, 'male', -1 / 3) for c in ['SF', 'NM', 'NF']))
    c3 = make_vec('SM', 'male', 1) - make_vec('SF', 'female', 1)
    c4 = make_vec('SM', 'female', 1) - make_vec('SM', 'male', 1)

    return {
        'H1': contrast(c1),
        'H2': contrast(c2),
        'H3': contrast(c3),
        'H4': contrast(c4),
    }


def _single_sim(args):
    return fit_and_test(simulate_dataset(*args))


def estimate_power_parallel(n_per_cond, sims, alpha, tau, sigma, d, workers=None):
    seeds = np.random.randint(0, 1e8, size=sims)
    args = [(n_per_cond, tau, sigma, d, s) for s in seeds]
    counts = {'H1': 0, 'H2': 0, 'H3': 0, 'H4': 0}

    with ProcessPoolExecutor(max_workers=workers) as executor:
        for res in tqdm(executor.map(_single_sim, args), total=sims, desc=f"Npc={n_per_cond}"):
            for h, p in res.items():
                if p < alpha:
                    counts[h] += 1

    return {h: counts[h] / sims for h in counts}


def main():
    total_range = range(100, 1501, 100)
    power_data = []

    for total in total_range:
        n_per_cond = total // 4
        pw = estimate_power_parallel(
            n_per_cond=n_per_cond,
            sims=1000,
            alpha=0.05,
            tau=0.5,
            sigma=1.0,
            d=0.2
        )
        pw['total'] = total
        power_data.append(pw)

    df_pw = pd.DataFrame(power_data)

    # Plot power curves
    plt.figure()
    for h in ['H1', 'H2', 'H3', 'H4']:
        plt.plot(df_pw['total'], df_pw[h], label=h)
    plt.axhline(0.8, linestyle='--')
    plt.xlabel('Total sample size')
    plt.ylabel('Power')
    plt.title('Power vs. Total N')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Print minimum N achieving ≥80% across all hypotheses
    meets = df_pw[df_pw[['H1', 'H2', 'H3', 'H4']].ge(0.8).all(axis=1)]
    print("Minimum total N with ≥80% power for all hypotheses:", meets.total.min())

# Minimum total N with ≥80% power for all hypotheses: 1300

if __name__ == "__main__":
    main()

