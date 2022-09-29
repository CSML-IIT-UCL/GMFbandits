from abc import ABC
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm


class FairBanditProblem:
    def __init__(self, mu_star, n_arms, n_groups, true_rewards):
        self.mu_star = mu_star
        self.n_arms = n_arms
        self.n_groups = n_groups
        self.true_rewards = np.sort(true_rewards, axis=0)

    def get_noisy_reward(self, x, a):
        raise NotImplementedError

    def get_reward(self, x, a):
        raise NotImplementedError

    @staticmethod
    def get_rewards_ecdfs_from_rewards(rewards, rs):
        indic = (rewards <= rs).astype(float)
        return np.mean(indic, axis=0)

    def get_context(self):
        raise NotImplementedError

    def get_rewards_ecdfs(self, X_hist, s_hist, X, s, mu):
        rewards = np.zeros(self.n_arms)
        rs = np.zeros(self.n_arms)
        for i in range(self.n_arms):
            rewards[i] = np.einsum("jk,k->j", X_hist[s_hist == s[i]], mu)
            rs[i] = np.dot(X[i], mu)
        return self.get_rewards_ecdfs_from_rewards(rewards, rs)

    def get_cdfs_estimate(self, rs, s):
        cdfs = np.zeros(self.n_arms)
        for i in range(self.n_arms):
            cdfs[i] = np.searchsorted(
                self.true_rewards[:, s[i]], rs[i], side="right"
            ) / len(self.true_rewards[:, s[i]])

        # assert np.equal(cdfs, self.get_rewards_ecdfs_from_rewards(self.true_rewards, rs)).all()
        return cdfs


class OnlineRidge:
    def __init__(self, reg_param, d):
        self.reg_param = reg_param
        # self.X = []
        # self.y = []
        self.Ireg = np.eye(d) * reg_param
        self.XTX = reg_param * np.eye(d)
        self.XTXinv = np.eye(d) / reg_param
        self.XTy = np.zeros(d)
        self.theta = np.zeros(d)
        self.updates_count = 0
        self.beta = 0

    def update(self, X, y):
        self.updates_count += 1
        coeff = 1 + np.dot(X, self.XTXinv @ X)
        xtx = np.outer(X, X)
        self.XTX = self.XTX + xtx
        self.XTXinv -= (self.XTXinv @ xtx @ self.XTXinv) / coeff
        self.XTy += y * X
        self.beta = np.sqrt(
            2 * np.log(np.linalg.det(self.XTX) / np.linalg.det(self.Ireg))
        )

        self.theta = self.XTXinv @ self.XTy

    def predict(self, X):
        return np.dot(X, self.theta)


class Policy:
    def select_arm(self, X, s):
        raise NotImplementedError

    def update_history(self, X, r, a, s):
        pass

    def get_mu_estimate(self):
        return 0


class RidgePolicy(Policy, ABC):
    def __init__(self, reg_param, d):
        self.online_ridge = OnlineRidge(reg_param, d)
        self.reg_param = reg_param
        self.d = d

    def get_mu_estimate(self):
        return self.online_ridge.theta


class OraclePolicy(Policy, ABC):
    def __init__(self, P: FairBanditProblem):
        self.P = P

    def get_mu_estimate(self):
        return self.P.mu_star


class Random(Policy):
    def select_arm(self, X, s):
        n_arms = len(X[:, 0])
        return np.random.randint(low=0, high=n_arms)


class Optimal(OraclePolicy):
    def __init__(self, reg_param, d, P):
        super().__init__(P)

    def select_arm(self, X, s):
        n_arms = len(X[:, 0])
        est_rewards = [np.dot(X[a], self.P.mu_star) for a in range(n_arms)]
        arg_max = np.argwhere(est_rewards == np.max(est_rewards))[0, :]
        return np.random.choice(arg_max)
        # return np.random.randint(low=0, high=n_arms)


class OptFair(OraclePolicy):
    def __init__(self, P: FairBanditProblem):
        super().__init__(P)

    def select_arm(self, X, s):
        rs = np.einsum("jk,k->j", X, self.P.mu_star)
        cdfs = self.P.get_cdfs_estimate(rs, s)
        return np.argmax(cdfs), dict(cdfs=cdfs)


class Greedy(RidgePolicy):
    def select_arm(self, X, s):
        n_arms = len(X[:, 0])
        est_rewards = [np.dot(X[a], self.online_ridge.theta) for a in range(n_arms)]
        arg_max = np.argwhere(est_rewards == np.max(est_rewards))[0, :]
        return np.random.choice(arg_max)
        # return np.random.randint(low=0, high=n_arms)

    def update_history(self, X, r, a, s):
        self.online_ridge.update(X[a], r)


class FairGreedy(RidgePolicy):
    def __init__(self, reg_param, d, mu_noise_level):
        super().__init__(reg_param, d)
        self.mu_noise_level = mu_noise_level
        self.t = 0
        self.t0 = 0
        self.ecdf_groups = []
        self.ecdf_contexts = []
        self.actions = []
        self.rewards = []

    def select_arm(self, X, s):
        n_arms = len(X[:, 0])
        self.t += 1
        if self.t < 3:
            return np.random.randint(low=0, high=(n_arms - 1))
        mu_hat = self.online_ridge.theta + (
            self.mu_noise_level / np.sqrt(self.d * self.t)
        ) * np.random.randn(self.d)

        # Compute ECDF
        X_hist = np.concatenate([c[None, :] for c in self.ecdf_contexts])
        s_hist = np.array(self.ecdf_groups)
        rewards = np.zeros(n_arms)
        ecdfs = np.zeros(n_arms)
        for i in range(n_arms):
            rewards = np.einsum("jk,k->j", X_hist[s_hist == s[i]], mu_hat)
            rs = np.dot(X[i], mu_hat)
            indic = (rewards <= rs).astype(float)
            ecdfs[i] = np.mean(indic)

        nan = np.isnan(ecdfs)
        if any(nan):
            ecdfs[nan] = np.random.rand(sum(nan))

        # Select Arm
        arg_max = np.argwhere(ecdfs == np.max(ecdfs))[0, :]
        return np.random.choice(arg_max)

    def update_history(self, X, r, a, s):
        n_arms = len(X[:, 0])
        t0_new = np.floor((self.t - 1) / 2)
        if self.t0 != t0_new:
            self.online_ridge.update(
                self.ecdf_contexts[self.actions[0]], self.rewards[0]
            )
            self.ecdf_contexts, self.actions, self.rewards = (
                self.ecdf_contexts[n_arms:],
                self.actions[1:],
                self.rewards[1:],
            )
            self.ecdf_groups = self.ecdf_groups[n_arms:]
        for i in range(n_arms):
            self.ecdf_contexts.append(X[i])
            self.ecdf_groups.append(s[i])
        self.actions.append(a)
        self.rewards.append(r)

        self.t0 = t0_new


class FairGreedyKnownCDF(RidgePolicy):
    def __init__(self, reg_param, d, noise_magnitude, P: FairBanditProblem):
        super().__init__(reg_param, d)
        self.P = P
        self.noise_magnitude = noise_magnitude

    def select_arm(self, X, s):
        t = self.online_ridge.updates_count + 1
        mu_hat = self.online_ridge.theta + (
            self.noise_magnitude / np.sqrt(t)
        ) * np.random.randn(self.d)
        rs = np.einsum("jk,k->j", X, mu_hat)
        est_cdfs = self.P.get_cdfs_estimate(rs, s)
        arg_max = np.argwhere(est_cdfs == np.max(est_cdfs))[0, :]
        return np.random.choice(arg_max)
        # return np.random.randint(low=0, high=n_arms)

    def update_history(self, X, r, a, s):
        self.online_ridge.update(X[a], r)


class FairGreedyKnownMuStar(OraclePolicy):
    def __init__(self, P: FairBanditProblem):
        super().__init__(P)
        self.t = 0
        self.rewards = None
        self.groups = None

    def select_arm(self, X, s):
        n_arms = len(X[:, 0])
        self.t += 1
        if self.t < 2:
            return np.random.randint(low=0, high=(n_arms - 1))

        # Compute ECDF
        ecdfs = np.zeros(n_arms)
        for i in range(n_arms):
            rs = np.dot(X[i], self.P.mu_star)
            indic = (self.rewards[self.groups == s[i]] <= rs).astype(float)
            ecdfs[i] = np.mean(indic, axis=0)

        nan = np.isnan(ecdfs)
        if any(nan):
            ecdfs[nan] = np.random.rand(sum(nan))

        # Select Arm
        arg_max = np.argwhere(ecdfs == np.max(ecdfs))[0, :]
        return np.random.choice(arg_max)

    def update_history(self, X, r, a, s):
        rs = np.einsum("jk,k->j", X, self.P.mu_star)
        if self.rewards is None:
            self.rewards = rs
            self.groups = s

        self.rewards = np.concatenate((self.rewards, np.array(rs)))
        self.groups = np.concatenate((self.groups, np.array(s)))


class FairGreedyNoNoise(FairGreedy):
    def __init__(self, reg_param, d):
        super().__init__(reg_param, d, mu_noise_level=0)


class OFUL(RidgePolicy):
    def __init__(self, reg_param, d, expl_coeff=1.0):
        super().__init__(reg_param, d)
        self.ec = expl_coeff

    def get_reward_ucb(self, X):
        n_arms = len(X[:, 0])
        t = self.online_ridge.updates_count + 1
        est_rewards = [np.dot(X[a], self.online_ridge.theta) for a in range(n_arms)]

        for i in range(n_arms):
            # Confidence Bound wrt direction
            ucb = (
                self.ec
                * self.online_ridge.beta
                * (
                    (
                        np.dot(
                            np.transpose(X[i]), np.dot(self.online_ridge.XTXinv, X[i])
                        )
                        * np.log(t)
                    )
                    ** 0.5
                )
            )
            est_rewards[i] = est_rewards[i] + ucb

        return np.concatenate([e[None] for e in est_rewards])

    def select_arm(self, X, s):
        rewards_ucb = self.get_reward_ucb(X)
        arg_max = np.argwhere(rewards_ucb == np.max(rewards_ucb))[0, :]
        return np.random.choice(arg_max)
        # return np.random.randint(low=0, high=n_arms)

    def update_history(self, X, r, a, s):
        self.online_ridge.update(X[a], r)


def test_policy(
    policy_gen: callable, P: FairBanditProblem, T=100, compute_true_cdf=True, seed=None
):
    policy = policy_gen()
    policy_name = policy.__class__.__name__
    print(f"Testing {policy_name}")
    np.random.seed(seed)
    history = defaultdict(list)
    if hasattr(P, "generate_context"):
        P.generate_context(n=T)
    if hasattr(P, "reset"):
        P.reset()

    for t in tqdm(range(T)):
        X, s = P.get_context()
        a = policy.select_arm(X, s)
        r = P.get_noisy_reward(X[a], a)
        policy.update_history(X=X, r=r, a=a, s=s)

        history["round"].append(t + 1)
        history["actions"].append(a)
        history["groups"].append(s)

        history["sel_group"].append(s[a])
        history["contexts"].append(X)
        history["rewards"].append(r)

        # quantities for evaluation
        exp_rewards = [P.get_reward(X[a1], a1) for a1 in range(P.n_arms)]
        # mu_star_rewards = [np.dot(P.mu_star, X[a1]) for a1 in range(P.n_arms)]
        history["exp_rewards"].append(exp_rewards)
        history["exp_reward_policy"].append(exp_rewards[a])
        history["exp_reward_opt"].append(max(exp_rewards))

        if compute_true_cdf:
            # rs = np.einsum("jk,k->j", X, P.mu_star)
            fair_rewards = P.get_cdfs_estimate(exp_rewards, s)
            history["fair_rewards"].append(fair_rewards)
            history["fair_reward_policy"].append(fair_rewards[a])
            history["fair_reward_opt"].append(max(fair_rewards))

    res = pd.DataFrame(history)
    res["pseudo_instant_regret"] = res["exp_reward_opt"] - res["exp_reward_policy"]
    res["pseudo_instant_fair_regret"] = (
        res["fair_reward_opt"] - res["fair_reward_policy"]
    )
    res["pseudo_regret"] = res["pseudo_instant_regret"].cumsum()
    res["pseudo_fair_regret"] = res["pseudo_instant_fair_regret"].cumsum()
    res["policy"] = policy_name
    res["T"] = T

    return policy, pd.DataFrame(res)
