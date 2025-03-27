import numpy as np

import torch


def apply_dirichlet_noise(action_probs, dirichlet_alpha=0.3, exploration_fraction=0.25):
    noise = np.random.dirichlet([dirichlet_alpha] * len(action_probs))
    return (1 - exploration_fraction) * action_probs + exploration_fraction * noise


class MonteCarloTreeSearch:

    def __init__(self, game_state, nnet):
        """
        Parameters
        ----------
        game_state : gym.Env
        nnet       : torch.nn.Module
        """
        self.game_state = game_state
        self.nnet = nnet
        self._quality_states_actions = {}
        self._state_number_of_visits_actions = {}
        self._state_number_of_visits = {}
        self._is_terminal_states = {}
        self._state_legal_actions = {}
        self._state_action_probs = {}

        self._current_player = {}
        self._previous_player = {}

    def reset(self):
        self._quality_states_actions = {}
        self._state_number_of_visits_actions = {}
        self._state_number_of_visits = {}
        self._is_terminal_states = {}
        self._state_legal_actions = {}
        self._state_action_probs = {}

        self._current_player = {}
        self._previous_player = {}

    def _get_action_probabilities(self, state):
        legal_actions = self._state_legal_actions[state]
        total_visits = sum(
            self._state_number_of_visits_actions.get((action, state), 1e-8)
            for action in legal_actions
        )
        action_probs = {
            action: self._state_number_of_visits_actions.get((action, state), 1e-8)
            / total_visits
            for action in legal_actions
        }
        return action_probs

    def _expand(self, state, game_state, game_obs):
        self._state_legal_actions[state] = game_state.action_space.legal_actions
        self._state_number_of_visits[state] = 1e-8
        self._current_player[state] = game_state.current_player
        self._previous_player[state] = game_state.previous_player

        player_at_leaf = self._current_player[state]

        # 1 current player wins, -1 previous player wins
        with torch.no_grad():
            action_probs, predicted_outcome = self.nnet(game_obs[0])
            action_probs = action_probs.cpu().numpy().flatten()
            predicted_outcome = predicted_outcome.item()

        self._state_action_probs[state] = action_probs
        return player_at_leaf, predicted_outcome

    def _adjust_result(self, state, predicted_outcome, player_at_leaf):
        if player_at_leaf == self._current_player[state]:
            return predicted_outcome
        if player_at_leaf == self._previous_player[state]:
            return -predicted_outcome
        return 0

    def _simulate(self, game_state, game_obs, root=False, c_param=1.4):
        state = game_state.get_string_representation()

        if state not in self._is_terminal_states:
            self._is_terminal_states[state] = game_state.game_result()

        if self._is_terminal_states[state] is not None:
            # terminal node
            winner = self._is_terminal_states[state]
            predicted_outcome = 1.0
            return winner, predicted_outcome

        if state not in self._state_legal_actions:
            return self._expand(state, game_state, game_obs)

        best_action, best_ucb = self.best_action(state, root=root, c_param=c_param)

        # Traverse to next node in tree
        game_state.skip_next_human_render()
        observation, reward, terminated, truncated, info = game_state.step(best_action)
        player_at_leaf, predicted_outcome = self._simulate(
            game_state=game_state, game_obs=observation
        )

        # result is -1 if previous player won, and 1 if current player won.
        result = self._adjust_result(state, predicted_outcome, player_at_leaf)

        self._backpropagate(state, best_action, result)
        return player_at_leaf, predicted_outcome

    def _backpropagate(self, state, best_action, result):
        if (best_action, state) in self._quality_states_actions:
            q_old = self._quality_states_actions[(best_action, state)]
            self._quality_states_actions[(best_action, state)] = q_old + result
            self._state_number_of_visits_actions[(best_action, state)] += 1
        else:
            self._quality_states_actions[(best_action, state)] = result
            self._state_number_of_visits_actions[(best_action, state)] = 1

        self._state_number_of_visits[state] += 1

    def best_action(self, state, root=False, c_param=1.4):
        best_ucb = -np.inf
        best_action = None

        N = self._state_number_of_visits[state]
        LOGN = np.log(N)

        action_probs = self._state_action_probs[state].copy()
        if root: # add dirichlet noise
            action_probs = apply_dirichlet_noise(action_probs)

        for action in self._state_legal_actions[state]:
            if (action, state) in self._quality_states_actions:
                p = action_probs[action]
                q = self._quality_states_actions[(action, state)]
                n = self._state_number_of_visits_actions[(action, state)]
                ucb = (q / n) + c_param * p * np.sqrt(LOGN / n)
            else:
                ucb = np.inf

            if ucb > best_ucb:
                best_ucb = ucb
                best_action = action

        return best_action, best_ucb

    def search(self, init_state, init_obs, num_simulations=1000):
        for itr in range(num_simulations):
            self.game_state.set_string_representation(init_state)
            self._simulate(self.game_state, game_obs=init_obs, root=True)

        self.game_state.set_string_representation(init_state)

        action_probs = self._get_action_probabilities(init_state)
        best_action = max(action_probs, key=action_probs.get)
        return best_action, action_probs
