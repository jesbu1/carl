from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import numpy as np



class Agent:
    """An general class for RL agents.
    """

    def sample(self, horizon, policy, record=False, env=None, mode='train'):
        """Samples a rollout from the agent.

        Arguments:
            horizon: (int) The length of the rollout to generate from the agent.
            policy: (policy) The policy that the agent will use for actions.
            record: (bool) Whether to record the rollout
            env: (gym.Env) Environment to rollout
            mode: (str) Whether we're training or testing, used for resetting the env

        Returns: (dict) A dictionary containing data from the rollout.
        """
        times, rewards = [], []
        policy.mode = mode
        O, A, reward_sum, done = [env.reset(mode=mode)], [], 0, False
        policy.reset()
        for t in range(horizon):
            start = time.time()
            policy_action = policy.act(O[t], t)
            A.append(policy_action)
            times.append(time.time() - start)
            obs, reward, done, info = env.step(policy_action) # A[t]
            O.append(obs)
            reward_sum += reward
            rewards.append(reward)
            if done:
                break
        if record:
            env.close()
        print("Average action selection time: ", np.mean(times))
        print("Rollout length: ", len(A))
        return {
            "obs": np.array(O),
            "ac": np.array(A),
            "reward_sum": reward_sum,
            "rewards": np.array(rewards),
            "catastrophe": info['Catastrophe'],
        }
