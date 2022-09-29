import plot
from plot import *
from policies import *
from pathlib import Path


def main():
    exp_prefix = "trial_"
    mode = "manual_reward_2"
    # context_mode = 'centered_uniform'
    context_mode = "standard_uniform"
    reg_param = 0.1  # regularization parameter for the ridge regression
    T = 50  # total number of rounds
    plot_mult = 1.2  # 0.8, higher value gives a larger plot

    for n_arms in (4,):
        exp_dir = (
            f"exps/simulation/{exp_prefix}compare_all_{mode}/K={n_arms}_T={T}_lambda{reg_param}"
            f"_CM{context_mode}/"
        )
        exp_dir = run(
            n_arms=n_arms,
            exp_dir=exp_dir,
            plot_mult=plot_mult,
            mode=mode,
            context_mode=context_mode,
            T=T,
            reg_param=reg_param,
        )

        plot.main(dir=f"{exp_dir}plots/", mult=plot_mult, mode_histogram="percentage")


def run(
    # problem parameters,
    problem_seed=5,
    n_arms=2,
    noise_magnitude=2.0,
    # noise_magnitude = .0
    n_samples_cdf=int(1e6),
    group_bias=16.0,
    mu_noise_level=1e-8,
    compute_density=False,
    context_mode="centered_uniform",
    # algo parameters
    reg_param=0.1,
    T=500,
    # T = 5000,
    algo_seeds=tuple(range(10)),
    mode="normal_reward",
    plot_mult=0.8,
    exp_dir="exps/simu_simple/",
):

    Path(f"{exp_dir}plots/").mkdir(parents=True, exist_ok=True)

    np.random.seed(problem_seed)

    P = SimuFB(
        mode=mode,
        n_arms=n_arms,
        group_bias=group_bias,
        noise_magnitude=noise_magnitude,
        n_samples_cdf=n_samples_cdf,
        context_mode=context_mode,
    )

    params = dict(
        d=P.d,
        n_arms=n_arms,
        noise_magnitude=noise_magnitude,
        n_samples_cdf=n_samples_cdf,
    )

    plot.set_figsize_dpi(figsize=[plot_mult * i for i in (4, 3.5)], dpi=200)

    # Test if satisfies the assumptions
    nu = np.einsum("ijk,j->ik", P.B, P.mu_star)
    nu_norm = np.linalg.norm(nu, axis=1)
    muTc = np.einsum("ij,j->i", P.c, P.mu_star)
    print("nu (should be != 0):", nu)
    print("nu_norm (should be != 0):", nu_norm)
    print("muTc (can be also 0)", muTc)

    # plot rewards histograms
    mode_plot_rewards = "density" if compute_density else "hist"
    plot_rewards(
        P.true_rewards, mode=mode_plot_rewards, save=True, dir=f"{exp_dir}plots/"
    )

    policies_generators = [
        lambda: Random(),
        lambda: OFUL(reg_param, P.d),
        lambda: FairGreedy(reg_param, P.d, mu_noise_level),
        # lambda: Greedy(reg_param, P.d),
        # lambda: FairOFUL(reg_param, P.d),
        # lambda: FairGreedyKnownCDF(reg_param, P.d, mu_noise_level, P),
        # lambda: FairGreedyNoNoise(reg_param, P.d),
        # lambda: FairOFULKnownCDF(reg_param, P.d, P),
        # lambda: FairGreedyKnownMuStar(P),
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


class SimuFB(FairBanditProblem):
    def __init__(
        self,
        mode="normal_reward",
        context_mode="centered_uniform",
        n_arms=2,
        group_bias=16.0,
        noise_magnitude=2.0,
        n_samples_cdf=int(1e6),
    ):
        # d = (2 + 2) * n_arms + 1
        self.d = (2 + 2) * n_arms + 1
        self.da = (self.d - 1) // n_arms
        self.context_mode = context_mode
        self.noise_magnitude = noise_magnitude
        self.contexts = None
        self.context_count = None

        self.B, self.c = np.zeros((n_arms, self.d, self.da)), np.zeros((n_arms, self.d))
        for i in range(n_arms):
            if self.context_mode == "centered_uniform":
                self.c[i, self.d - 1] = (i * group_bias) / (
                    n_arms - 1
                ) - 0.5 * group_bias
            else:
                self.c[i, self.d - 1] = (i * group_bias) / (n_arms - 1)
            self.B[i, i * self.da : (i + 1) * self.da] = 1.0 * np.eye(self.da)

        if mode == "normal_reward":
            mu_star = np.random.randn(self.d)
            mu_star[-1] = 1  # bias for groups, later specified by c
            # mu_star = np.concatenate([i*np.random.rand(d//n_arms) for i in np.linspace(-1, 1, num=n_arms)])

        elif mode == "manual_reward":
            reward_scale = 1.0
            base_group_mu = (0.4, 0.3)
            if len(base_group_mu) > self.da:
                raise NotImplementedError
            mu_star = np.zeros(self.d)
            for i in range(n_arms):
                mu_star[i * self.da : i * self.da + 2] = [
                    reward_scale * (i + 1) * j for j in base_group_mu
                ]
            mu_star[-1] = 1  # bias for groups, later specified by c

        elif mode == "manual_reward_2":
            reward_scale = 10.0
            # group_mus = [(.4, .3), (.1, .9), (.5, .5), (.2, .2)]
            group_mus = [
                (0.4, 0.3, 0.7),
                (0.8,),
                (0.5, 0.5),
                (0.2, 0.2, 0.2, 0.2),
                (0.04, 0.09),
                (0.09, 0.08, 0.09, 0.02),
                (0.02,),
                (0.07, 0.08, 0.02),
            ]
            mu_star = np.zeros(self.d)
            for i in range(n_arms):
                mu_star[i * self.da : i * self.da + len(group_mus[i])] = [
                    reward_scale * j for j in group_mus[i]
                ]
            mu_star[-1] = 1  # bias for groups, later specified by c

        elif mode == "same_manual_reward":
            reward_scale = 10.0
            base_group_mu = (0.4, 0.3)
            if len(base_group_mu) > self.da:
                raise NotImplementedError
            mu_star = np.zeros(self.d)
            for i in range(n_arms):
                mu_star[i * self.da : i * self.da + 2] = [
                    reward_scale * m for m in base_group_mu
                ]
            mu_star[-1] = 1  # bias for groups, later specified by c

            for i in range(n_arms):
                self.c[i, self.d - 1] = 0
        else:
            raise NotImplementedError

        self.n_arms = n_arms
        self.contexts_samples = self.generate_context(n=n_samples_cdf)
        self.generate_context(n=1)
        true_rewards = np.einsum("ijk,k->ij", self.contexts_samples, mu_star)
        super().__init__(n_arms=n_arms, mu_star=mu_star, true_rewards=true_rewards)

    def generate_context(self, n=1):
        if self.context_mode == "standard_uniform":
            Y = np.random.rand(n, self.n_arms, self.da)
        elif self.context_mode == "centered_uniform":
            Y = np.random.rand(n, self.n_arms, self.da) - 0.5
        else:
            raise NotImplementedError

        X = np.einsum("ijk,lik->lij", self.B, Y) + self.c[None, :, :]
        self.contexts = X
        self.context_count = 0
        return X

    def get_context(self):
        if self.context_count >= self.contexts.shape[0]:
            self.generate_context(n=10)

        X = self.contexts[self.context_count]
        self.context_count += 1
        return X

    def get_noisy_reward(self, x, a):
        return np.dot(x, self.mu_star) + self.noise_magnitude * np.random.randn(1)[0]

    def get_reward(self, x, a):
        return np.dot(x, self.mu_star)


if __name__ == "__main__":
    main()
