import numpy as np
import torch

from .search import apply_dirichlet_noise


class FastMonteCarloTreeSearch:
    """Optimized MCTS implementation using vectorized operations."""

    def __init__(self, game_state, nnet):
        self.game_state = game_state
        self.nnet = nnet
        self._nodes = {}

    def reset(self):
        self._nodes = {}

    def _expand(self, state, env, obs):
        legal_actions = env.action_space.legal_actions
        with torch.no_grad():
            policy_logits, value = self.nnet(obs[0])
            policy = policy_logits.cpu().numpy().flatten()
            value = value.item()
        policy = policy[legal_actions]
        node = {
            "legal_actions": np.array(legal_actions, dtype=np.int32),
            "policy": policy,
            "Q": np.zeros(len(legal_actions), dtype=np.float32),
            "N": np.zeros(len(legal_actions), dtype=np.float32),
            "visits": 1e-8,
            "current_player": env.current_player,
            "previous_player": env.previous_player,
            "is_terminal": None,
        }
        self._nodes[state] = node
        return env.current_player, value

    def _best_action(self, node, root=False, c_param=1.4):
        policy = node["policy"]
        if root:
            policy = apply_dirichlet_noise(policy)
        log_n = np.log(node["visits"])
        Q = node["Q"]
        N = node["N"]
        with np.errstate(divide="ignore", invalid="ignore"):
            ucb = np.where(
                N > 0,
                (Q / N) + c_param * policy * np.sqrt(log_n / N),
                np.inf,
            )
        return int(np.argmax(ucb))

    def _backpropagate(self, path, player_at_leaf, value):
        for node, idx in path:
            if player_at_leaf == node["current_player"]:
                result = value
            elif player_at_leaf == node["previous_player"]:
                result = -value
            else:
                result = 0
            node["Q"][idx] += result
            node["N"][idx] += 1
            node["visits"] += 1

    def _simulate(self, env, obs, root=False, c_param=1.4):
        path = []
        first = True
        while True:
            state = env.get_string_representation()
            node = self._nodes.get(state)
            if node is None:
                result = env.game_result()
                if result is not None:
                    self._nodes[state] = {
                        "legal_actions": [],
                        "policy": None,
                        "Q": np.array([]),
                        "N": np.array([]),
                        "visits": 1e-8,
                        "current_player": env.current_player,
                        "previous_player": env.previous_player,
                        "is_terminal": result,
                    }
                    player = result
                    value = 1.0
                else:
                    player, value = self._expand(state, env, obs)
                break
            if node["is_terminal"] is None:
                node["is_terminal"] = env.game_result()
            if node["is_terminal"] is not None:
                player = node["is_terminal"]
                value = 1.0
                break
            best_idx = self._best_action(node, root=first and root, c_param=c_param)
            action = node["legal_actions"][best_idx]
            path.append((node, best_idx))
            env.skip_next_human_render()
            obs, reward, terminated, truncated, info = env.step(action)
            first = False
        self._backpropagate(path, player, value)
        return player, value

    def search(self, init_state, init_obs, num_simulations=1000):
        for _ in range(num_simulations):
            self.game_state.set_string_representation(init_state)
            self._simulate(self.game_state, init_obs, root=True)
        self.game_state.set_string_representation(init_state)
        node = self._nodes[init_state]
        counts = node["N"]
        probs = counts / counts.sum()
        action_probs = {
            action: probs[i] for i, action in enumerate(node["legal_actions"])
        }
        best_action = node["legal_actions"][int(np.argmax(counts))]
        return best_action, action_probs
