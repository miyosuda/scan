# SCAN

## About

Replicating SCAN algorithm described in Google DeepMind's paper ["SCAN: Learning Abstract Hierarchical Compositional Visual Concepts"](https://arxiv.org/abs/1707.03389)



Image datasets are created with [Rodent envrironment. ](https://github.com/miyosuda/rodent/tree/master/examples/04_texture_replace)

## Requirements

- Tensorflow 1.2 or later
- Python2 or 3


## How to train

First extract dataset, and then run `main.py`

```
$ tar xvf data.tar.gz
$ python main.py
```

## Result

### Sym2img

Symbol to image conversion result.

1) Generated images when `wall_color=white` is specified.

![](doc/sym2img/img0.png)

2) Generated images when `wall_color=white`, `floor_color=white` are specified.

![](doc/sym2img/img1.png)

3) Generated images when `wall_color=white`, `floor_color=white`, `obj_color=white` are specified.

![](doc/sym2img/img2.png)

4) Generated images when `wall_color=white`, `floor_color=white`, `obj_color=white`, `obj_id=ice_lolly` are specified.

![](doc/sym2img/img3.png)

### Img2sym

|  Input                    |  Output                                                                        |
|---------------------------|--------------------------------------------------------------------------------|
| ![](doc/img2sym/img0.png) | `obj_color=white`, `wall_color=white`, `floor_color=white`, `obj_id=ice_lolly` |

(All of the outputs are correct.)

|  Input                    |  Output                                                                  |
|---------------------------|--------------------------------------------------------------------------|
| ![](doc/img2sym/img1.png) | `obj_color=purple`, `wall_color=dark_yellow`, `obj_id=hat`               |

(Correct `obj_color` was `red`, but confused as `purple`. `floor_color` was not specifiled in the output.)


### βVAE disentanglement

Disentanglement result for latent variables for object parameters. (Wall color, Object color, Floor Color, Object Type, Object position).

![](doc/disentangle/z8.png) Wall color

![](doc/disentangle/z10.png) Obj color

![](doc/disentangle/z14.png) Floor color

![](doc/disentangle/z0.png) Obj Type

![](doc/disentangle/z18.png) Obj pos (and Obj Type)




