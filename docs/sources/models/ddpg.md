<span style="float:right;">[[source]](https://github.com/pyhashzero/pyhashzero/blob/master/hz/ai/models/ddpg.py#L17)</span>
### DDPGModel

```python
hz.ai.models.ddpg.DDPGModel.hz.ai.models.ddpg.DDPGModel(lr=0.001, tau=0.0001, gamma=0.99, actor=None, critic=None)
```


Deep Deterministic Policy Gradient

__Arguments__

- __actor_model__ (`keras.nn.Model` instance): See [Model](#) for details.
- __critic_model__ (`keras.nn.Model` instance): See [Model](#) for details.
- __optimizer__ (`keras.optimizers.Optimizer` instance):
See [Optimizer](#) for details.
- __action_inp__ (`keras.layers.Input` / `keras.layers.InputLayer` instance):
See [Input](#) for details.
- __tau__ (float): tau.
- __gamma__ (float): gamma.

----

<span style="float:right;">[[source]](https://github.com/pyhashzero/pyhashzero/blob/master/hz/ai/models/ddpg.py#L52)</span>

### load


```python
DDPGModel.load(self, path='')
```

----

<span style="float:right;">[[source]](https://github.com/pyhashzero/pyhashzero/blob/master/hz/ai/models/ddpg.py#L58)</span>

### save


```python
DDPGModel.save(self, path='')
```

----

<span style="float:right;">[[source]](https://github.com/pyhashzero/pyhashzero/blob/master/hz/ai/models/ddpg.py#L114)</span>

### predict


```python
DDPGModel.predict(self, state=None)
```

----

<span style="float:right;">[[source]](https://github.com/pyhashzero/pyhashzero/blob/master/hz/ai/models/ddpg.py#L118)</span>

### train


```python
DDPGModel.train(self, batch=None, update_target=True)
```

----

<span style="float:right;">[[source]](https://github.com/pyhashzero/pyhashzero/blob/master/hz/ai/models/ddpg.py#L156)</span>

### evaluate


```python
DDPGModel.evaluate(self)
```
