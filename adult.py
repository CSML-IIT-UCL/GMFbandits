import plot
from data import *
from plot import *
from policies import *


def main():
    exp_prefix = "trial_"
    group = "SEX"
    density = 1  # 1 to load the full folktables dataset, smaller than one for a smaller portion.
    n_samples_per_group = 500000  # smaller groups are discarded
    T = 5000  # total number of rounds
    n_seeds = 10  # number of runs of each policy (same data)
    reg_param = 0.1  # regularization parameter for the ridge regression
    noise_magnitude = None  # None to use the true rewards from the dataset as noisy
    expl_coeff_oful = 0.1  # OFUL exploration coefficient. O is equivalent to Greedy.
    alpha = 0.01  # Parameter for a fair policy that can be used as a baseline

    plot_mult = 1.2  # higher value gives a larger plot
    exp_dir = (
        f"exps/adult_new/{exp_prefix}g_{group}_d_{density}_n_{n_samples_per_group}_nm{noise_magnitude}"
        f"_lambda_{reg_param}_T={T}_ns_{n_seeds}_ecOFUL{expl_coeff_oful}_alpha{alpha}/"
    )

    run(
        exp_dir=exp_dir,
        plot_mult=plot_mult,
        group=group,
        density=density,
        n_samples_per_group=n_samples_per_group,
        reg_param=reg_param,
        noise_magnitude=noise_magnitude,
        algo_seeds=tuple(range(n_seeds)),
        expl_coeff_oful=expl_coeff_oful,
        T=T,
        compute_density=True,
        alpha=alpha,
    )

    plot.main(dir=f"{exp_dir}plots/", mult=plot_mult, mode_histogram="percentage")


def run(
    # problem parameters,
    problem_seed=42,
    group_bias=16.0,
    mu_noise_level=1e-8,
    compute_density=False,
    group="SEX",
    density=1,
    n_samples_per_group=50000,
    poly_degree=1,
    noise_magnitude=None,
    alpha=0.1,
    # algo parameters
    reg_param=0.1,
    expl_coeff_oful=1.0,
    T=500,
    # T = 5000,
    algo_seeds=tuple(range(10)),
    plot_mult=0.8,
    exp_dir="exps/adult/simple/",
):
    from pathlib import Path

    Path(f"{exp_dir}plots/").mkdir(parents=True, exist_ok=True)

    P = load_adult(
        group=group,
        density=density,
        poly_degree=poly_degree,
        n_samples_per_group=n_samples_per_group,
        seed=problem_seed,
        noise_magnitude=noise_magnitude,
    )
    n_arms = P.n_arms

    params = dict(
        d=P.d,
        n_arms=n_arms,
        group=group,
        n_samples_per_group=n_samples_per_group,
        poly_degree=poly_degree,
        density=density,
    )

    plot.set_figsize_dpi(figsize=[plot_mult * i for i in (4, 3.5)], dpi=200)

    # plot rewards histograms
    mode_plot_rewards = "density" if compute_density else "hist"
    plot_rewards(
        P.true_rewards,
        mode=mode_plot_rewards,
        save=True,
        dir=f"{exp_dir}plots/",
        suffix="adult_",
    )

    policies_generators = [
        lambda: Random(),
        lambda: OFUL(reg_param, P.d, expl_coeff_oful),
        lambda: FairGreedy(reg_param, P.d, mu_noise_level),
        lambda: Greedy(reg_param, P.d),
        # lambda: FairGreedyKnownCDF(reg_param, P.d, mu_noise_level, P),
        # lambda: FairGreedyKnownMuStar(P)
        # lambda: FairGreedyNoNoise(reg_param, P.d),
        # lambda: FairOFUL(reg_param, P.d, expl_coeff_oful),
        # lambda: FairOFULKnownCDF(reg_param, P.d, P, expl_coeff_oful),
        # lambda: FairLearnGreedy(
        #     reg_param, P.d, r=np.ones(n_arms) / n_arms, alpha=T * alpha
        # ),
        # lambda: FairLearnOFUL(reg_param, P.d, r=np.ones(n_arms)/n_arms, alpha=T*alpha, expl_coeff=expl_coeff_oful),
    ]

    total_ps, total_dfs = [], []
    for policy_gen in policies_generators:
        ps, dfs = [], []
        for s in algo_seeds:
            np.random.seed(s)
            p, results = test_policy(policy_gen, P=P, T=T)
            results["seed"] = s
            for k, v in params.items():
                results[k] = v

            dfs.append(results)
            ps.append(p)
        # df_dict[policy_class.__name__] = dfs
        policy_name = dfs[0]["policy"].drop_duplicates()[0]

        print(f"Results for {policy_name}")
        for df in dfs:
            df.hist(
                column="actions", bins=n_arms,
            )
            break

        plt.title(f"action_histo_{policy_name}")
        plt.show()
        for m in ["pseudo_regret", "pseudo_fair_regret"]:
            for df in dfs:
                df[m].plot()
            plt.title(f"{m}_{policy_name}")
            plt.show()
        for p in ps:
            print(f"mu_est_MSE = {np.mean((p.get_mu_estimate() - P.mu_star) ** 2)}")
            print(f"mu_est = {p.get_mu_estimate()}")
            print(f"mu_star = {P.mu_star}")
            break

        for p in ps:
            print(f"mu_est_MSE = {np.mean((p.get_mu_estimate() - P.mu_star) ** 2)}")
            print(f"mu_est = {p.get_mu_estimate()}")
            print(f"mu_star = {P.mu_star}")
            break

        total_ps.extend(ps)
        total_dfs.extend(dfs)

    result_df = pd.concat(total_dfs)
    result_df.to_csv(f"{exp_dir}results.csv")

    return exp_dir


if __name__ == "__main__":
    main()
