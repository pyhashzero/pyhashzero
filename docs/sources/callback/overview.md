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
