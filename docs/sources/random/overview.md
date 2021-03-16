<span style="float:right;">[[source]](https://github.com/pyhashzero/pyhashzero/blob/master/hz/ai/core.py#L272)</span>
### CoreRandom

```python
hz.ai.core.CoreRandom.hz.ai.core.CoreRandom()
```


Abstract base class for all implemented random processes.

Do not use this abstract base class directly but instead
use one of the concrete random processes implemented.

To implement your own random processes,
you have to implement the following methods:

- `decay`
- `sample`
- `reset`

----

<span style="float:right;">[[source]](https://github.com/pyhashzero/pyhashzero/blob/master/hz/ai/core.py#L290)</span>

### reset


```python
CoreRandom.reset(self)
```



Reset random state.

----

<span style="float:right;">[[source]](https://github.com/pyhashzero/pyhashzero/blob/master/hz/ai/core.py#L296)</span>

### decay


```python
CoreRandom.decay(self)
```



decay

----

<span style="float:right;">[[source]](https://github.com/pyhashzero/pyhashzero/blob/master/hz/ai/core.py#L302)</span>

### sample


```python
CoreRandom.sample(self)
```



Sample random state.

__Returns__

- __sample__ (abstract): Random state.
