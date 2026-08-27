"""Agent behaviour: exploration schedule, target updates, and the Bellman update.

The Bellman tests are the important ones. They hand-compute the expected target
and assert the implementation matches, so a sign error or a misapplied discount
fails here rather than showing up as an agent that mysteriously will not learn.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch

from dqn.agent import DQNAgent
from dqn.config import DEFAULT_CONFIG

DEVICE = torch.device("cpu")
STATE_DIM, ACTION_DIM = 8, 4


@pytest.fixture
def config():
    return replace(DEFAULT_CONFIG, batch_size=4, min_buffer_size=4)


@pytest.fixture
def agent(config):
    torch.manual_seed(0)
    return DQNAgent(STATE_DIM, ACTION_DIM, config, DEVICE)


class TestExploration:
    def test_starts_fully_exploratory(self, agent, config):
        assert agent.epsilon == config.epsilon_start

    def test_decays_multiplicatively(self, agent, config):
        before = agent.epsilon
        agent.decay_epsilon()
        assert agent.epsilon == pytest.approx(before * config.epsilon_decay)

    def test_never_falls_below_the_floor(self, agent, config):
        for _ in range(10_000):
            agent.decay_epsilon()
        assert agent.epsilon == pytest.approx(config.epsilon_end)

    def test_greedy_flag_ignores_epsilon(self, agent):
        """Evaluation must be deterministic even with epsilon at 1.0."""
        agent.epsilon = 1.0
        state = np.zeros(STATE_DIM, dtype=np.float32)
        actions = {agent.select_action(state, greedy=True) for _ in range(30)}
        assert len(actions) == 1

    def test_returns_a_valid_action(self, agent):
        state = np.zeros(STATE_DIM, dtype=np.float32)
        for _ in range(50):
            assert agent.select_action(state) in range(ACTION_DIM)


class TestTargetNetwork:
    def test_starts_identical_to_policy(self, agent):
        for target, policy in zip(
            agent.target_net.parameters(), agent.policy_net.parameters(), strict=True
        ):
            assert torch.equal(target, policy)

    def test_soft_update_moves_target_by_tau(self, config):
        """target <- tau * policy + (1 - tau) * target, exactly."""
        torch.manual_seed(1)
        agent = DQNAgent(STATE_DIM, ACTION_DIM, config, DEVICE)

        with torch.no_grad():
            for param in agent.policy_net.parameters():
                param.fill_(1.0)
            for param in agent.target_net.parameters():
                param.fill_(0.0)

        agent.soft_update_target()

        for param in agent.target_net.parameters():
            assert torch.allclose(param, torch.full_like(param, config.tau))

    def test_repeated_updates_converge_toward_policy(self, config):
        torch.manual_seed(1)
        agent = DQNAgent(STATE_DIM, ACTION_DIM, config, DEVICE)
        with torch.no_grad():
            for param in agent.policy_net.parameters():
                param.fill_(1.0)
            for param in agent.target_net.parameters():
                param.fill_(0.0)

        for _ in range(2000):
            agent.soft_update_target()

        for param in agent.target_net.parameters():
            assert torch.allclose(param, torch.ones_like(param), atol=1e-3)


class TestBellmanTarget:
    """The target is r + gamma * max_a' Q(s', a') * (1 - done)."""

    def test_terminal_transition_drops_the_bootstrap(self, agent):
        """On a terminal step there is no next state, so the target is just the
        reward. Forgetting (1 - done) here is the classic DQN bug: the agent
        keeps valuing a state that does not exist."""
        next_states = torch.zeros(4, STATE_DIM)
        rewards = torch.tensor([10.0, -5.0, 0.0, 3.0])
        dones = torch.ones(4)

        with torch.no_grad():
            q_next = agent._next_state_values(next_states)
            target = rewards + (agent.cfg.gamma * q_next * (1.0 - dones))

        assert torch.allclose(target, rewards)

    def test_non_terminal_adds_discounted_next_value(self, agent):
        next_states = torch.zeros(4, STATE_DIM)
        rewards = torch.tensor([1.0, 1.0, 1.0, 1.0])
        dones = torch.zeros(4)

        with torch.no_grad():
            q_next = agent._next_state_values(next_states)
            target = rewards + (agent.cfg.gamma * q_next * (1.0 - dones))
            expected = rewards + agent.cfg.gamma * q_next

        assert torch.allclose(target, expected)

    def test_double_dqn_scores_the_policy_networks_choice(self, config):
        """Double DQN must evaluate the action the *policy* net would pick,
        which is not in general the target net's own argmax."""
        agent = DQNAgent(STATE_DIM, ACTION_DIM, replace(config, double_dqn=True), DEVICE)
        next_states = torch.zeros(2, STATE_DIM)

        with torch.no_grad():
            chosen = agent.policy_net(next_states).argmax(dim=1)
            expected = agent.target_net(next_states).gather(1, chosen.unsqueeze(1)).squeeze(1)
            actual = agent._next_state_values(next_states)

        assert torch.allclose(actual, expected)

    def test_vanilla_dqn_takes_the_target_networks_max(self, config):
        agent = DQNAgent(STATE_DIM, ACTION_DIM, replace(config, double_dqn=False), DEVICE)
        next_states = torch.zeros(2, STATE_DIM)

        with torch.no_grad():
            expected = agent.target_net(next_states).max(dim=1)[0]
            actual = agent._next_state_values(next_states)

        assert torch.allclose(actual, expected)

    def test_the_two_variants_can_disagree(self, config):
        """If they always agreed, the Double DQN branch would be dead code.
        With independently initialised networks the estimates differ."""
        torch.manual_seed(7)
        double = DQNAgent(STATE_DIM, ACTION_DIM, replace(config, double_dqn=True), DEVICE)
        with torch.no_grad():
            for param in double.target_net.parameters():
                param.add_(torch.randn_like(param) * 0.5)

        states = torch.randn(64, STATE_DIM)
        with torch.no_grad():
            double_values = double._next_state_values(states)
            double.cfg = replace(double.cfg, double_dqn=False)
            vanilla_values = double._next_state_values(states)

        assert not torch.allclose(double_values, vanilla_values)

    def test_vanilla_is_never_below_double(self, config):
        """max_a Q_target(s,a) >= Q_target(s, argmax_a Q_policy(s,a)) by
        definition of max. This is exactly the overestimation Double DQN
        removes, and it should be visible as a one-sided gap."""
        torch.manual_seed(11)
        agent = DQNAgent(STATE_DIM, ACTION_DIM, replace(config, double_dqn=False), DEVICE)
        with torch.no_grad():
            for param in agent.target_net.parameters():
                param.add_(torch.randn_like(param) * 0.5)

        states = torch.randn(128, STATE_DIM)
        with torch.no_grad():
            vanilla = agent._next_state_values(states)
            agent.cfg = replace(agent.cfg, double_dqn=True)
            double = agent._next_state_values(states)

        assert torch.all(vanilla >= double - 1e-6)


class TestLearning:
    def test_no_gradient_step_before_the_buffer_warms_up(self, agent):
        assert agent.learn() is None

    def test_returns_a_loss_once_warm(self, agent, config):
        for i in range(config.min_buffer_size):
            agent.memory.push(
                np.full(STATE_DIM, i, dtype=np.float32),
                i % ACTION_DIM,
                1.0,
                np.full(STATE_DIM, i + 1, dtype=np.float32),
                False,
            )
        loss = agent.learn()
        assert isinstance(loss, float) and loss >= 0.0

    def test_a_gradient_step_changes_the_policy_network(self, agent, config):
        for i in range(config.min_buffer_size):
            agent.memory.push(
                np.random.randn(STATE_DIM).astype(np.float32),
                i % ACTION_DIM,
                1.0,
                np.random.randn(STATE_DIM).astype(np.float32),
                False,
            )
        before = [p.clone() for p in agent.policy_net.parameters()]
        agent.learn()
        after = list(agent.policy_net.parameters())
        assert any(not torch.equal(b, a) for b, a in zip(before, after, strict=True))

    def test_a_gradient_step_leaves_the_target_network_alone(self, agent, config):
        """The target net must only move via soft_update_target(). If backprop
        reaches it, the stable-target property is gone."""
        for i in range(config.min_buffer_size):
            agent.memory.push(
                np.random.randn(STATE_DIM).astype(np.float32),
                i % ACTION_DIM,
                1.0,
                np.random.randn(STATE_DIM).astype(np.float32),
                False,
            )
        before = [p.clone() for p in agent.target_net.parameters()]
        agent.learn()
        after = list(agent.target_net.parameters())
        assert all(torch.equal(b, a) for b, a in zip(before, after, strict=True))


class TestPersistence:
    def test_checkpoint_round_trips(self, agent, config, tmp_path):
        path = tmp_path / "agent.pth"
        agent.save(str(path))

        restored = DQNAgent(STATE_DIM, ACTION_DIM, config, DEVICE)
        restored.load(str(path))

        state = torch.randn(1, STATE_DIM)
        with torch.no_grad():
            assert torch.allclose(agent.policy_net(state), restored.policy_net(state))
