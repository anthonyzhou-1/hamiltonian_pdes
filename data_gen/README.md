## Data Generation for 2D Shallow Water Equations

An implementation using [PyClaw](https://www.clawpack.org/pyclaw/). To generate data, setup PyClaw then run:

```
python swe_2d.py 
```

This will generate each data sample in a subfolder in ``outdir``. Initial conditions are manually set in the script. To assemble the subfolders into a single h5 dataset, run:

```
python assemble.py
```

This will read PyClaw outputs from each subfolder and create a h5 dataset. 