<span style="float:right;">[[source]](https://github.com/pyhashzero/pyhashzero/blob/master/hz/ai/core.py#L312)</span>
### CoreModel

```python
hz.ai.core.CoreModel.hz.ai.core.CoreModel()
```


Abstract base class for all implemented nn.

Do not use this abstract base class directly
but instead use one of the concrete nn implemented.

To implement your own nn, you have to implement the following methods:

- `act`
- `replay`
- `load`
- `save`

----

<span style="float:right;">[[source]](https://github.com/pyhashzero/pyhashzero/blob/master/hz/ai/core.py#L330)</span>

### load


```python
CoreModel.load(self)
```



load

----

<span style="float:right;">[[source]](https://github.com/pyhashzero/pyhashzero/blob/master/hz/ai/core.py#L336)</span>

### save


```python
CoreModel.save(self)
```



save

----

<span style="float:right;">[[source]](https://github.com/pyhashzero/pyhashzero/blob/master/hz/ai/core.py#L342)</span>

### predict


```python
CoreModel.predict(self)
```



Get the action for given state.

Accepts a state and returns an abstract action.

__Arguments__

- __state__ (abstract): Current state of the environment.

__Returns__

- __action__ (abstract): Network's predicted action for given state.

----

<span style="float:right;">[[source]](https://github.com/pyhashzero/pyhashzero/blob/master/hz/ai/core.py#L356)</span>

### train


```python
CoreModel.train(self)
```



Train the nn with given batch.

__Arguments__

- __batch__ (abstract): Mini Batch from Experience Replay Memory.

----

<span style="float:right;">[[source]](https://github.com/pyhashzero/pyhashzero/blob/master/hz/ai/core.py#L365)</span>

### evaluate


```python
CoreModel.evaluate(self)
```



evaluate
