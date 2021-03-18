import copy
import gc
import random

import gym
import numpy as np
import torch.nn as nn
from PIL import Image

from hz.ai.callback import (
    CallbackList,
    Loader,
    Renderer,
    TrainLogger,
    ValidationLogger
)
from hz.ai.memory import RingMemory
from hz.ai.models import DQNModel
from hz.ai.policy import (
    EpsilonGreedyPolicy,
    GreedyQPolicy
)
from hz.ai.utility import (
    easy_range,
    to_tensor
)
from hz.env.chess import (
    Board,
    create_move_labels,
    Move
)


class RLProcessor:
    def __init__(self):
        super(RLProcessor, self).__init__()

    @staticmethod
    def process_batch(batch):
        states = []
        actions = []
        rewards = []
        next_states = []
        terminals = []
        for state, action, reward, next_state, terminal in batch:
            states.extend(state)
            actions.append(action)
            rewards.append(reward)
            next_states.extend(next_state)
            terminals.append(terminal)
        states = np.asarray(states).astype('float32')
        actions = np.asarray(actions).astype('float32')
        rewards = np.asarray(rewards).astype('float32')
        next_states = np.asarray(next_states).astype('float32')
        terminals = np.asarray(terminals).astype('float32')
        return states, actions, rewards, next_states, terminals

    @staticmethod
    def process_state(state):
        state = Image.fromarray(state)
        state = state.resize((84, 84))
        state = np.array(state)
        state = np.expand_dims(state, 0)
        state = np.transpose(state, (0, 3, 1, 2))
        state = state.astype('uint8')
        return state


class DQN(nn.Module):
    def __init__(self, in_features, out_features):
        super(DQN, self).__init__()

        self.extractor = nn.Sequential(
            nn.Conv2d(in_channels=in_features, out_channels=32, kernel_size=(8, 8), stride=(4, 4)),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(4, 4), stride=(2, 2)),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1)),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True)
        )
        self.predictor = nn.Sequential(
            nn.Linear(in_features=7 * 7 * 64, out_features=512),
            nn.BatchNorm1d(num_features=512),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=512, out_features=out_features),
            nn.Softmax()
        )

    def forward(self, x):
        x = self.extractor(x)
        x = x.view(x.size(0), -1)
        x = self.predictor(x)
        return x


class DQNTrainer:
    def __init__(self, environment, memory, processor, model, callbacks, test_policy, train_policy, move_labels):
        super(DQNTrainer, self).__init__()

        self.environment = environment
        self.memory = memory
        self.processor = processor
        self.model = model
        self.callbacks = callbacks
        self.test_policy = test_policy
        self.train_policy = train_policy
        self.move_labels = move_labels

    def final_state(self):
        pass

    def get_step(self, action):
        return Move.from_uci(self.move_labels[action])

    def get_action(self, state, policy):
        if policy.use_network():
            action = self.model.predict(to_tensor(state))
            action = self.move_labels[int(action)]
            action = Move.from_uci(action)
        else:
            if len(list(self.environment.generate_legal_moves())) == 0:
                import cv2
                cv2.waitKey()

            action = random.choice(list(self.environment.generate_legal_moves()))

        if not self.environment.is_legal(action):
            action = random.choice(list(self.environment.generate_legal_moves()))
            # action = MCTSGameController().get_next_move(self.environment, time_allowed=1)

        return self.move_labels.index(action.str())

    def train(self, batch_size=32, max_action=200, max_episode=12000, warmup=120000):
        total_steps = 0
        self.callbacks.on_agent_begin(**{
            'agent_headers': ['episode_number', 'action_number', 'episode_reward'],
            'network_headers': ['loss']
        })
        for episode_number in easy_range(1, max_episode):
            episode_reward = 0
            state = self.environment.reset()
            state = self.processor.process_state(state)
            self.callbacks.on_episode_begin(**{
                'episode_number': episode_number,
                'state': state
            })

            for action_number in easy_range(1, max_action):
                action = self.get_action(state, self.train_policy)
                self.callbacks.on_action_begin(**{
                    'episode_number': episode_number,
                    'action_number': action_number,
                    'state': state,
                    'action': action
                })
                step = self.get_step(action)
                next_state, reward, terminal, _ = self.environment.step(step)
                if not terminal:
                    _, r, _, _ = self.environment.step(random.choice(list(self.environment.generate_legal_moves())))
                    reward -= r
                next_state = self.processor.process_state(next_state)
                if action_number >= max_action:
                    terminal = True

                self.callbacks.on_action_end(**{
                    'episode_number': episode_number,
                    'action_number': action_number,
                    'state': state,
                    'action': action,
                    'reward': reward,
                    'terminal': terminal,
                    'next_state': next_state
                })

                # clipped_reward = np.clip(reward - 0.25, -1, 1)
                self.memory.remember((state, action, reward, next_state, terminal))

                if total_steps > warmup:
                    self.train_policy.decay()
                    if total_steps % batch_size == 0:
                        self.callbacks.on_replay_begin()
                        mini_batch = self.memory.sample()
                        batch = self.processor.process_batch(mini_batch)
                        loss = self.model.train(batch)

                        self.callbacks.on_replay_end(**{
                            'loss': loss
                        })

                episode_reward += reward
                state = copy.deepcopy(next_state)
                total_steps += 1

                if terminal or self.environment.is_game_over():
                    self.callbacks.on_episode_end(**{
                        'episode_number': episode_number,
                        'action_number': action_number,
                        'episode_reward': episode_reward
                    })
                    gc.collect()
                    break

        self.environment.close()
        self.callbacks.on_agent_end(**{
            'total_steps': total_steps
        })

    def evaluate(self, max_action=50, max_episode=12):
        total_steps = 0
        self.callbacks.on_agent_begin()
        for episode_number in easy_range(1, max_episode):
            episode_reward = 0
            state = self.environment.reset()
            state = self.processor.process_state(state)
            self.callbacks.on_episode_begin(**{
                'episode_number': episode_number,
                'state': state
            })

            for action_number in easy_range(1, max_action):
                action = self.get_action(state, self.test_policy)
                self.callbacks.on_action_begin(**{
                    'episode_number': episode_number,
                    'action_number': action_number,
                    'state': state,
                    'action': action
                })

                step = self.get_step(action)
                next_state, reward, terminal, _ = self.environment.step(step)
                next_state = self.processor.process_state(next_state)
                if action_number >= max_action:
                    terminal = True

                self.callbacks.on_action_end(**{
                    'episode_number': episode_number,
                    'action_number': action_number,
                    'state': state,
                    'action': action,
                    'reward': reward,
                    'terminal': terminal,
                    'next_state': next_state
                })

                episode_reward += reward
                state = copy.deepcopy(next_state)
                total_steps += 1

                if terminal:
                    self.callbacks.on_episode_end(**{
                        'episode_number': episode_number,
                        'action_number': action_number,
                        'episode_reward': episode_reward
                    })
                    gc.collect()
                    break

        self.environment.close()
        self.callbacks.on_agent_end(**{
            'total_steps': total_steps
        })


def create_env():
    try:
        environment = gym.make('Chess-v0')
    except Exception as ex:
        print(str(ex))
        environment = Board()
    return environment


def run():
    move_labels = create_move_labels()

    memory = RingMemory(batch_size=32)
    processor = RLProcessor()

    model = DQNModel(network=DQN(in_features=3, out_features=len(move_labels)))

    experiment = '.'
    case = '.'
    run_name = 'test'
    run_path = 'saves/01.dqn/{}/{}/{}/'.format(experiment, case, run_name)

    environment = create_env()
    agent = DQNTrainer(
        environment=environment,
        memory=memory,
        processor=processor,
        model=model,
        callbacks=CallbackList([
            TrainLogger(run_path=run_path, interval=10),
            Loader(model=model, run_path=run_path, interval=10),
            Renderer(environment=environment)
        ]),
        train_policy=EpsilonGreedyPolicy(min_value=0.1),
        test_policy=GreedyQPolicy(),
        move_labels=move_labels
    )
    agent.train(max_episode=12000, warmup=120000, max_action=200, batch_size=32)

    environment = create_env()
    agent = DQNTrainer(
        environment=environment,
        memory=memory,
        processor=processor,
        model=model,
        callbacks=CallbackList([
            ValidationLogger(run_path=run_path, interval=1),
            Renderer(environment=environment),
        ]),
        train_policy=EpsilonGreedyPolicy(min_value=0.1),
        test_policy=GreedyQPolicy(),
        move_labels=move_labels
    )
    agent.evaluate(max_action=200)


if __name__ == '__main__':
    run()
