fail for v1.4 to use real data
modify ../scripts/tf_cnn_benchmarks/preprocessing.py
delete the following line
```python
# from tensorflow.contrib.data.python.ops import interleave_ops
```
