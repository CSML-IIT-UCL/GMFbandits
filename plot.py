import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def set_figsize_dpi(figsize, dpi):
    plt.rcParams["figure.figsize"] = figsize
    plt.rcParams["figure.dpi"] = dpi


def plot_rewards(
    rewards,
    mode="density",
    linestyles=("solid", "dashed", "dotted", "dashdot"),
    save=True,
    dir="",
    suffix="",
):
    n_arms = len(rewards[0])

    for cdf in (True, False):
        cdf_string = "CDF" if cdf else "PDF"
        for i in range(n_arms):
            if mode == "density":
                sns.kdeplot(
                    rewards[:, i],
                    label=f"Group {i + 1}",
                    linestyle=linestyles[i % len(linestyles)],
                    cumulative=cdf,
                )

            elif mode == "hist":
                plt.hist(
                    rewards[:, i], alpha=0.3, label=f"g{i}", bins=1000, cumulative=cdf
                )
            else:
                raise NotImplementedError

        plt.title(f"Rewards {cdf_string}")
        plt.ylabel("")

        plt.xlabel("Reward")
        plt.legend()
        plt.tight_layout()
        if save:
            plt.savefig(f"{dir}{suffix}true_rewards_{mode}_{cdf_string}.png")
            plt.savefig(f"{dir}{suffix}true_rewards_{mode}_{cdf_string}.pdf")
        plt.show()

    # figure(figsize=fig_size, dpi=dpi)
    # for i in range(n_arms):
    #     plt.hist((P.true_rewards[:, i] - muTc[i]) / nu_norm[i], alpha=0.3, label=f"g{i}", bins=1000)
    # plt.title("true normalized rewards histogram")
    # plt.xlabel("normalized rewards")
    # plt.legend()
    # plt.show()


def plot_results(
    result_df,
    title_dict=dict(
        pseudo_regret="Standard pseudo-regret", pseudo_fair_regret="Fair pseudo-regret"
    ),
    policy_dict=dict(
        Random="Uniform Random",
        OFUL="OFUL",
        FairGreedy="Fair-greedy",
        FairGreedyNoNoise="Fair-greedy no noise",
        FairGreedyKnownCDF="Fair-greedy (Oracle CDF)",
        FairGreedyKnownMuStar="Fair-greedy (Oracle rewards)",
    ),
    line_style_dict=dict(
        Random="dashed",
        OFUL="dotted",
        FairGreedy="solid",
        Greedy="dashdot",
        FairGreedyKnownCDF="dashed",
        FairGreedyKnownMuStar="dashed",
    ),
    save_fig=True,
    dir="",
    selected_policies=None,
    y_lim_dict={},
    x_lim_dict={},
    prefix_name="",
    mode_histogram="percentage",  # number or percentage
):
    selected_policies = (
        result_df["policy"].drop_duplicates().values
        if selected_policies is None
        else selected_policies
    )

    g_col = "sel_group" if "sel_group" in result_df.columns else "actions"
    n_groups = len(result_df[g_col].drop_duplicates().values)
    n_arms = len(result_df["actions"].drop_duplicates().values)

    hist_dict = {}
    if "pseudo_fair_regret" in x_lim_dict.keys():
        N_rounds = x_lim_dict["pseudo_fair_regret"][-1]
    else:
        N_rounds = None
    for p in selected_policies:
        policy_df = result_df.loc[result_df["policy"] == p]
        label = policy_dict[p] if p in policy_dict else p

        hist_dict[label] = []
        for seed in range(10):
            policy_df_0 = policy_df[policy_df["seed"] == seed]
            n_rounds_max = (
                N_rounds if N_rounds is not None else len(policy_df_0[g_col].values)
            )
            if "groups" in result_df.columns:
                all_groups = np.concatenate(
                    [
                        np.array([int(a) + 1 for a in b[1:-1].split(" ")])
                        for b in policy_df_0["groups"].values[:n_rounds_max]
                    ]
                )
            else:
                all_groups = np.concatenate(
                    [np.arange(1, n_groups + 1) for i in range(n_rounds_max)]
                )

            hist = np.histogram(
                policy_df_0[g_col].values[:n_rounds_max] + 1, bins=n_groups
            )[0]
            n_groups_total = np.histogram(all_groups, bins=n_groups)[0]

            if mode_histogram == "number":
                hist_dict[label].append(hist)
            elif mode_histogram == "percentage":
                hist_dict[label].append(100 * hist / n_groups_total)
            else:
                raise NotImplementedError

        hist_dict[label] = np.concatenate(
            [g[None, :] for g in hist_dict[label]], axis=0
        )
        # plt.hist(policy_df_0['sel_group'], histtype='step', label=label, alpha=0.5)

    # plt.legend()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    width = 0.2
    for i, (label, hist) in enumerate(hist_dict.items()):

        ax.bar(
            height=hist.mean(axis=0),
            x=np.array((range(1, n_groups + 1))) + width * i,
            yerr=hist.std(axis=0),
            width=width,
            label=label,
        )
        ax.set_xticks(np.array((range(1, n_groups + 1))) + width * 1.7)

    # add some
    if mode_histogram == "number":
        ax.set_title(f"# of selected groups at T={N_rounds}")
        plt.yscale("log")
    elif mode_histogram == "percentage":
        ax.set_title(f"Percentage of selected groups at T={N_rounds}")
        plt.axhline(y=100.0 / n_arms, linestyle="dotted", color="black")
    else:
        raise NotImplementedError

    ax.set_xticklabels((f"G{i+1}" for i in range(n_groups)))
    # ax.legend((rects1[0], rects2[0]), ('Men', 'Women'))
    plt.legend()
    if save_fig:
        plt.savefig(f"{dir}{prefix_name}{mode_histogram}hist_group.png")
        plt.savefig(f"{dir}{prefix_name}{mode_histogram}hist_group.pdf")
    plt.show()

    for m in [
        "pseudo_regret",
        "pseudo_fair_regret",
    ]:
        for p in selected_policies:
            policy_df = result_df.loc[result_df["policy"] == p]
            policy_df_seeds = policy_df.groupby(by=["round"], as_index=False)
            mean_df = policy_df_seeds.mean()
            std_df = policy_df_seeds.std()

            rounds = mean_df["round"]
            metric_mean = mean_df[m]
            metric_std = std_df[m]

            label = policy_dict[p] if p in policy_dict else p
            linestyle = line_style_dict[p] if p in line_style_dict else None

            plt.plot(rounds, metric_mean, label=label, linestyle=linestyle)
            plt.fill_between(
                rounds, metric_mean - metric_std, metric_mean + metric_std, alpha=0.3
            )

        title = title_dict[m] if m in title_dict else m
        plt.title(title)
        plt.xlabel("# of rounds")
        if m in y_lim_dict:
            plt.ylim(y_lim_dict[m])
        if m in x_lim_dict:
            plt.xlim(x_lim_dict[m])
        plt.legend()
        plt.tight_layout()
        if save_fig:
            plt.savefig(f"{dir}{prefix_name}{m}.png")
            plt.savefig(f"{dir}{prefix_name}{m}.pdf")
        plt.show()
        plt.close()


def main_adult(dir="", mult=1.2, x_dim=4, y_dim=3.5):
    selected = [
        "Random",
        "OFUL",
        "FairGreedy",
        "Greedy",
        # 'FairGreedyKnownCDF', 'FairGreedyKnownMuStar'
    ]
    y_lim_dict = dict(pseudo_regret=[-1, 120], pseudo_fair_regret=[-1, 60])
    x_lim_dict = dict(pseudo_regret=[-5, 2500], pseudo_fair_regret=[-5, 2500])
    prefix_name = "adult_"
    main(
        dir=dir,
        mult=mult,
        dpi=200,
        save_fig=True,
        selected_policies=selected,
        y_lim_dict=y_lim_dict,
        prefix_name=prefix_name,
        x_lim_dict=x_lim_dict,
        x_dim=x_dim,
        y_dim=y_dim,
    )


def main(mult=0.8, dpi=200, save_fig=True, dir="", x_dim=4, y_dim=3.5, **kwargs):
    set_figsize_dpi(figsize=[mult * i for i in (x_dim, y_dim)], dpi=dpi)
    import pandas as pd

    result_df = pd.read_csv(f"{dir}../results.csv")

    plot_results(result_df, save_fig=save_fig, dir=dir, **kwargs)


if __name__ == "__main__":
    # main_adult(dir=f'exps/adult_multi/g_RAC1P_na_10_d_1_n_5000_pd_1_nm0.2_lambda_0.01_T=5000_ns_10_ecOFUL0.01/plots/',
    #            y_dim=2.7,x_dim=3.0, mult=1.0)
    main_adult(
        dir=f"exps/adult_multi/g_RAC1P_na_10_d_1_n_5000_pd_1_nm0.2_lambda_0.01_T=5000_ns_10_ecOFUL0.01/plots/"
    )
    # for d in os.listdir("exps/adult_new"):
    #     main(dir=f'exps/adult_new/{d}/plots/')
    # main()
