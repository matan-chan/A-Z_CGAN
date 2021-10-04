# A-Z CGAN

image generator which generate letters images


## usage
list of possible inputs for each letter \
0 #,
1 $,
2 &,
3 @,
4 a,
5 b,
6 c,
7 d,
8 e,
9 f,
10 g,
11 h,
12 i,
13 j,
14 k,
15 l,
16 m,
17 w,
18 p,
19 q,
20 r,
21 s,
22 t,
23 u,
24 v,
25 w,
26 x,
27 y,
28 z,
```python
from train import generate_letter
x=16
generate_letter(x)
```
the output image will be saved in the `new_predictions` folder


## training:
first run the `process_data()` function
then call the `train()` function.

```python
from train import train

train(epochs=40_000, batch_size=128, save_interval=500)
```

## example:
<p align="left">
  <img width="500" src="https://github.com/matan-chan/A-Z_CGAN/blob/main/examples/ex1.png?raw=true">
</p>





