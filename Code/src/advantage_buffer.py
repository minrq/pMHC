# This file is modified from https://github.com/cpnota/autonomous-learning-library/all/memory/generalized_advantage.py


class AdvantageBuffer:
    def __init__(self, args):
        self.config = args
        
        self._states = []
        self._actions = []
        self._rewards = []

    def store(self, states, actions, rewards):
        if states is None:
            return
        if not self._states:
            self._states = [states]
            self._actions = [actions]
            self._rewards = [rewards]

        elif len(self._states) <= self.config["nsteps"]:
            self._states.append(states)
            self._actions.append(actions)
            self._rewards.append(rewards)

        else:
            raise Exception("Buffer exceeded")

    def advantages(self):
        pass

    def _compute_advantages(self, td_errors):
        advantages = td_errors.clone()
        current_advantages = advantages[0] * 0

        # the final advantage is always 0
        advantages[-1] = current_advantages
        for i in range(self.n_steps):
            t = self.n_steps - 1 - i
            mask = self._states[t + 1].mask.float()
            current_advantages = td_errors[t] + self.config["gamma"] * self.config["lam"] * current_advantages * mask
            advantages[t] = current_advantages

        return advantages

    def _clear_buffers(self):
        self._states = []
        self._actions = []
        self._rewards = []

