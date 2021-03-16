### Introduction

---

<span style="float:right;">[[source]](https://github.com/pyhashzero/pyhashzero/blob/master/hz/ai/models/dqn.py#L19)</span>
### DQNModel

```python
hz.ai.models.dqn.DQNModel.hz.ai.models.dqn.DQNModel(lr=0.001, tau=0.0001, gamma=0.99, network=None)
```


Deep Q Network

__Arguments__

- __models__ (`keras.nn.Model` instance): See [Model](#) for details.
- __optimizer__ (`keras.optimizers.Optimizer` instance):
See [Optimizer](#) for details.
- __tau__ (float): tau.
- __gamma__ (float): gamma.

----

<span style="float:right;">[[source]](https://github.com/pyhashzero/pyhashzero/blob/master/hz/ai/models/dqn.py#L45)</span>

### load


```python
DQNModel.load(self, path='')
```

----

<span style="float:right;">[[source]](https://github.com/pyhashzero/pyhashzero/blob/master/hz/ai/models/dqn.py#L49)</span>

### save


```python
DQNModel.save(self, path='')
```

----

<span style="float:right;">[[source]](https://github.com/pyhashzero/pyhashzero/blob/master/hz/ai/models/dqn.py#L77)</span>

### predict


```python
DQNModel.predict(self, state=None)
```

----

<span style="float:right;">[[source]](https://github.com/pyhashzero/pyhashzero/blob/master/hz/ai/models/dqn.py#L81)</span>

### train


```python
DQNModel.train(self, batch=None, update_target=False)
```

----

<span style="float:right;">[[source]](https://github.com/pyhashzero/pyhashzero/blob/master/hz/ai/models/dqn.py#L112)</span>

### evaluate


```python
DQNModel.evaluate(self)
```


---
