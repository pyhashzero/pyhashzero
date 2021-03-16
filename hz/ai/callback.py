import os

import numpy as np

from hz.ai.core import CoreCallback


class CallbackList(CoreCallback):
    def __init__(self, callbacks):
        super(CallbackList, self).__init__()

        self.callbacks = callbacks

    def on_action_begin(self, *args, **kwargs):
        for callback in self.callbacks:
            callback.on_action_begin(*args, **kwargs)

    def on_action_end(self, *args, **kwargs):
        for callback in self.callbacks:
            callback.on_action_end(*args, **kwargs)

    def on_agent_begin(self, *args, **kwargs):
        for callback in self.callbacks:
            callback.on_agent_begin(*args, **kwargs)

    def on_agent_end(self, *args, **kwargs):
        for callback in self.callbacks:
            callback.on_agent_end(*args, **kwargs)

    def on_episode_begin(self, *args, **kwargs):
        for callback in self.callbacks:
            callback.on_episode_begin(*args, **kwargs)

    def on_episode_end(self, *args, **kwargs):
        for callback in self.callbacks:
            callback.on_episode_end(*args, **kwargs)

    def on_replay_begin(self, *args, **kwargs):
        for callback in self.callbacks:
            callback.on_replay_begin(*args, **kwargs)

    def on_replay_end(self, *args, **kwargs):
        for callback in self.callbacks:
            callback.on_replay_end(*args, **kwargs)


class Loader(CoreCallback):
    def __init__(self, model, run_path, interval):
        super().__init__()

        self.network = model
        self.run_path = run_path
        self.interval = interval

    def on_agent_begin(self, *args, **kwargs):
        # weights path should be run path
        self.network.load(self.run_path)
        self.network.save(self.run_path)

    def on_agent_end(self, *args, **kwargs):
        self.network.save(self.run_path)

    def on_episode_end(self, *args, **kwargs):
        if 'episode_number' in kwargs:
            if kwargs['episode_number'] % self.interval == 0:
                self.network.save(self.run_path)


class Renderer(CoreCallback):
    def __init__(self, environment):
        super().__init__()

        self.environment = environment

    def on_action_end(self, *args, **kwargs):
        self.environment.render()

    def on_episode_begin(self, *args, **kwargs):
        self.environment.render()


class TrainLogger(CoreCallback):
    def __init__(self, run_path, interval):
        super().__init__()

        self.run_path = run_path
        self.interval = interval

        if not os.path.exists(run_path):
            os.makedirs(run_path)

        self.agent_data_path = self.run_path + 'train-agent-data.csv'
        self.network_data_path = self.run_path + 'train-nn-data.csv'
        self.episode_end_message_raw = '\repisode {:05d} ended in {:04d} actions, total reward: {:+08.2f}'

    def on_agent_begin(self, *args, **kwargs):
        if 'agent_headers' in kwargs:
            with open(self.agent_data_path, 'w') as file:
                file.write(','.join(kwargs['agent_headers']) + '\n')

        if 'network_headers' in kwargs:
            with open(self.network_data_path, 'w') as file:
                file.write(','.join(kwargs['network_headers']) + '\n')

    def on_episode_begin(self, *args, **kwargs):
        pass

    def on_episode_end(self, *args, **kwargs):
        if 'episode_number' in kwargs and 'action_number' in kwargs and 'episode_reward' in kwargs:
            end = '\n' if kwargs['episode_number'] % self.interval == 0 else ''
            print(self.episode_end_message_raw.format(kwargs['episode_number'], kwargs['action_number'], kwargs['episode_reward']), end=end)
            with open(self.agent_data_path, 'a') as file:
                file.write(','.join(list(map(str, [kwargs['episode_number'], kwargs['action_number'], kwargs['episode_reward']]))) + '\n')

    def on_replay_end(self, *args, **kwargs):
        if 'loss' in kwargs:
            if not isinstance(kwargs['loss'], (list, tuple)):
                kwargs['loss'] = [kwargs['loss']]

            with open(self.network_data_path, 'a') as file:
                file.write(','.join(list(map(str, kwargs['loss']))) + '\n')


class ValidationLogger(CoreCallback):
    def __init__(self, run_path, interval):
        super().__init__()

        self.run_path = run_path
        self.interval = interval

        if not os.path.exists(run_path):
            os.makedirs(run_path)

        self.agent_data_path = self.run_path + 'test-agent-data.csv'
        self.episode_end_message_raw = '\repisode {:05d} ended in {:04d} actions, total reward: {:+08.2f}'

    def on_agent_begin(self, *args, **kwargs):
        with open(self.agent_data_path, 'w') as file:
            file.write('episode_number,action_number,episode_reward\n')

    def on_episode_begin(self, *args, **kwargs):
        pass

    def on_episode_end(self, *args, **kwargs):
        if 'episode_number' in kwargs and 'action_number' in kwargs and 'episode_reward' in kwargs:
            end = '\n' if kwargs['episode_number'] % self.interval == 0 else ''
            print(self.episode_end_message_raw.format(kwargs['episode_number'], kwargs['action_number'], kwargs['episode_reward']), end=end)
            with open(self.agent_data_path, 'a') as file:
                file.write(','.join(list(map(str, [kwargs['episode_number'], kwargs['action_number'], kwargs['episode_reward']]))) + '\n')


class Visualizer(CoreCallback):
    def __init__(self, model, predicate=lambda x: True):
        super().__init__()

        self.model = model
        self.predicate = predicate

    def on_action_begin(self, *args, **kwargs):
        inputs = []
        if 'state' in kwargs:
            inputs.append(np.expand_dims(kwargs['state'], axis=0))

        if len(self.model.inputs) == 2 and 'action' in kwargs:
            inputs.append(np.expand_dims(kwargs['action'], axis=0))
