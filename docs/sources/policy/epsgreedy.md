<span style="float:right;">[[source]](https://github.com/pyhashzero/pyhashzero/blob/master/hz/ai/policy.py#L20)</span>
### EpsilonGreedyPolicy

```python
hz.ai.policy.EpsilonGreedyPolicy.hz.ai.policy.EpsilonGreedyPolicy(max_value=1.0, min_value=0.0, decay_steps=1)
```


Epsilon Greedy

__Arguments__

- __max_value__ (float): .
- __min_value__ (float): .
- __decay_steps__ (int): .

----

<span style="float:right;">[[source]](https://github.com/pyhashzero/pyhashzero/blob/master/hz/ai/policy.py#L37)</span>

### reset


```python
EpsilonGreedyPolicy.reset(self)
```

----

<span style="float:right;">[[source]](https://github.com/pyhashzero/pyhashzero/blob/master/hz/ai/policy.py#L40)</span>

### decay


```python
EpsilonGreedyPolicy.decay(self)
```

----

<span style="float:right;">[[source]](https://github.com/pyhashzero/pyhashzero/blob/master/hz/ai/policy.py#L44)</span>

### use_network


```python
EpsilonGreedyPolicy.use_network(self)
```
