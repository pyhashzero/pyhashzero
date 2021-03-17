## Available Callbacks

| Name                                                  | Implementation                         |
|-------------------------------------------------------|----------------------------------------|
| [GifMaker](/callback/gif-maker)                       | `hz.ai.callback.GifMaker`            |
| [TrainLogger](/callback/train-logger)                 | `hz.ai.callback.TrainLogger`         |
| [TestLogger](/callback/test-logger)                   | `hz.ai.callback.TestLogger`          |
| [LayerVisualizer](/callbacks/layer-visualizer)        | `hz.ai.callback.LayerVisualizer`     |
| [WeightLoader](/callback/weight-loader)               | `hz.ai.callback.WeightLoader`        |
| [EnvironmentRenderer](/callback/environment-renderer) | `hz.ai.callback.EnvironmentRenderer` |

---

## Common API

All callbacks share a common API. This allows you to use different callbacks.

<span style="float:right;">[[source]](https://github.com/pyhashzero/pyhashzero/blob/master/hz/ai/core.py#L92)</span>
### CoreCallback

```python
hz.ai.core.CoreCallback.hz.ai.core.CoreCallback()
```


Abstract base class for all implemented callback.

Do not use this abstract base class directly but instead use one of the concrete callback implemented.

To implement your own callback, you have to implement the following methods:

- `on_action_begin`
- `on_action_end`
- `on_replay_begin`
- `on_replay_end`
- `on_episode_begin`
- `on_episode_end`
- `on_agent_begin`
- `on_agent_end`

----

<span style="float:right;">[[source]](https://github.com/pyhashzero/pyhashzero/blob/master/hz/ai/core.py#L137)</span>

### on_action_begin


```python
CoreCallback.on_action_begin(self)
```



Called at beginning of each agent action

----

<span style="float:right;">[[source]](https://github.com/pyhashzero/pyhashzero/blob/master/hz/ai/core.py#L143)</span>

### on_action_end


```python
CoreCallback.on_action_end(self)
```



Called at end of each agent action

----

<span style="float:right;">[[source]](https://github.com/pyhashzero/pyhashzero/blob/master/hz/ai/core.py#L113)</span>

### on_agent_begin


```python
CoreCallback.on_agent_begin(self)
```



Called at beginning of each agent play

----

<span style="float:right;">[[source]](https://github.com/pyhashzero/pyhashzero/blob/master/hz/ai/core.py#L119)</span>

### on_agent_end


```python
CoreCallback.on_agent_end(self)
```



Called at end of each agent play

----

<span style="float:right;">[[source]](https://github.com/pyhashzero/pyhashzero/blob/master/hz/ai/core.py#L125)</span>

### on_episode_begin


```python
CoreCallback.on_episode_begin(self)
```



Called at beginning of each environment episode

----

<span style="float:right;">[[source]](https://github.com/pyhashzero/pyhashzero/blob/master/hz/ai/core.py#L131)</span>

### on_episode_end


```python
CoreCallback.on_episode_end(self)
```



Called at end of each environment episode

----

<span style="float:right;">[[source]](https://github.com/pyhashzero/pyhashzero/blob/master/hz/ai/core.py#L149)</span>

### on_replay_begin


```python
CoreCallback.on_replay_begin(self)
```



Called at beginning of each nn replay

----

<span style="float:right;">[[source]](https://github.com/pyhashzero/pyhashzero/blob/master/hz/ai/core.py#L155)</span>

### on_replay_end


```python
CoreCallback.on_replay_end(self)
```



Called at end of each nn replay

