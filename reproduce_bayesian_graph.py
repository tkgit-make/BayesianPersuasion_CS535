from pathlib import Path
import csv
import os
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "math535-mpl"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import expit, logit
from scipy.stats import binom

SEED = 535
OUTPUT = Path(__file__).resolve().parent / "bayesian_graphs_output"
ACCURACY = 0.70
STATE_PRIOR = 0.30
ROUNDS = 30
SEQUENCES = 10_000
PERSUASION_TRIALS = 200_000
PRIOR_PAIRS = [(0.45, 0.55), (0.30, 0.70), (0.20, 0.80)]


def write_csv(filename, rows):
    with (OUTPUT / filename).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save(fig, name):
    fig.savefig(OUTPUT / f"{name}.png", dpi=220, bbox_inches="tight")
    fig.savefig(OUTPUT / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)


def common_prior_figure():
    """One sender/receiver: receiver approves at posterior >= 1/2.

    Sender wants approval in either state. With approval at indifference,
    V(p)=min(2p,1). One optimal policy has alpha=P(H|good)=1 and
    beta=P(H|bad)=min(p/(1-p),1). This is a designed experiment, distinct
    from the fixed-accuracy repeated observations used in figure 2.
    """
    rng = np.random.default_rng(SEED)
    priors = np.linspace(0, 1, 21)
    rows = []
    for p in priors:
        beta = p / (1-p) if p < 0.5 else 1.0
        optimal = min(2*p, 1.0)
        good = rng.random(PERSUASION_TRIALS) < p
        signal_h = rng.random(PERSUASION_TRIALS) < np.where(good, 1.0, beta)
        # At p=0 the H message has zero probability, so no posterior is needed.
        mass_h = p + (1-p)*beta
        if mass_h > 0:
            assert p / mass_h >= 0.5 - 1e-12
        assert np.isclose(mass_h, optimal)
        simulated = signal_h.mean()
        se = np.sqrt(optimal*(1-optimal)/PERSUASION_TRIALS)
        assert abs(simulated-optimal) <= 6*se + 6/PERSUASION_TRIALS
        rows.append(dict(prior=p, alpha=1.0, beta=beta, optimal=optimal,
                         full_disclosure=p, no_information=int(p >= 0.5),
                         monte_carlo=simulated, standard_error=se))
    write_csv("common_prior.csv", rows)

    dense = np.linspace(0, 1, 1001)
    beta = np.ones_like(dense)
    low = dense < 0.5
    beta[low] = dense[low]/(1-dense[low])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), layout="constrained")
    ax = axes[0]
    ax.plot(dense, np.minimum(2*dense, 1), color="#176b87", lw=2.4,
            label="Optimal experiment")
    ax.plot(dense, dense, "--", color="#c87840", label="Full disclosure")
    ax.step([0, 0.5, 1], [0, 1, 1], where="post", color="#777777",
            linestyle=":", label="No information")
    ax.scatter(priors, [r["monte_carlo"] for r in rows], s=13,
               color="#176b87", label="Monte Carlo", zorder=4)
    ax.set(title="Sender benefit from information design",
           ylabel="Probability of approval / sender payoff")
    ax.legend(frameon=False, fontsize=9)
    axes[1].plot(dense, beta, color="#176b87", lw=2.4)
    axes[1].set(title="One optimal experiment, with α = 1",
                ylabel="P(approval signal | bad state)")
    for ax in axes:
        ax.set(xlabel="Common prior P(good)", xlim=(0, 1), ylim=(-0.025, 1.04))
        ax.grid(alpha=0.15)
        ax.spines[["top", "right"]].set_visible(False)
    save(fig, "common_prior_information_design")


def repeated_observations_figure():
    """Same fresh signal for both receivers each round; state stays fixed.

    P(H|1)=r, P(H|0)=1-r. After t rounds with k positive observations,
    L_i,t=logit(p_i,0)+(2k-t)*logit(r). The exact mean averages the
    resulting probability gap over the mixture of the two binomials.
    """
    rng = np.random.default_rng(SEED)
    theta = rng.random(SEQUENCES) < STATE_PRIOR
    evidence = np.zeros((SEQUENCES, ROUNDS+1))
    for t in range(1, ROUNDS+1):
        h = rng.random(SEQUENCES) < np.where(theta, ACCURACY, 1-ACCURACY)
        evidence[:, t] = evidence[:, t-1] + (2*h.astype(int)-1)*logit(ACCURACY)
    rows = []
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
    for ax, (lo, hi) in zip(axes, PRIOR_PAIRS):
        gaps = expit(logit(hi)+evidence)-expit(logit(lo)+evidence)
        simulated = gaps.mean(axis=0)
        se = gaps.std(axis=0, ddof=1)/np.sqrt(SEQUENCES)
        exact = []
        for t in range(ROUNDS+1):
            k = np.arange(t+1)
            mass = (STATE_PRIOR*binom.pmf(k, t, ACCURACY)
                    +(1-STATE_PRIOR)*binom.pmf(k, t, 1-ACCURACY))
            z = (2*k-t)*logit(ACCURACY)
            value = mass @ (expit(logit(hi)+z)-expit(logit(lo)+z))
            exact.append(value)
            assert abs(simulated[t]-value) <= 6*se[t]+6/SEQUENCES
            rows.append(dict(prior_low=lo, prior_high=hi, round=t,
                             simulated_gap=simulated[t], exact_gap=value,
                             standard_error=se[t]))
        times = np.arange(ROUNDS+1)
        ax.plot(times, simulated, color="#2563eb", lw=2,
                label="Bayesian simulation")
        ax.plot(times, exact, color="#475569", ls="--", lw=1.3,
                label="Exact Bayesian mean")
        ax.axhline(hi-lo, color="#64748b", ls=":", label="Initial gap")
        ax.text(ROUNDS-0.4, simulated[-1]+0.02, f"{simulated[-1]:.4f}",
                color="#2563eb", ha="right", fontsize=9)
        ax.set(title=f"Priors {lo:.2f} / {hi:.2f}", xlabel="Signal round",
               xlim=(0, ROUNDS), ylim=(0, 1.02))
        ax.set_xticks(np.arange(0, ROUNDS+1, 5))
        ax.grid(axis="y", color="#e2e8f0")
        ax.spines[["top", "right"]].set_visible(False)
        print(f"Priors {lo:.2f}/{hi:.2f}: round {ROUNDS} exact gap {exact[-1]:.6f}")
    axes[0].set_ylabel("Mean absolute probability gap")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center",
               ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    save(fig, "bayesian_repeated_observations")
    write_csv("repeated_observations.csv", rows)


if __name__ == "__main__":
    OUTPUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    common_prior_figure()
    repeated_observations_figure()
    print(f"Saved both figures and their data to {OUTPUT}")

