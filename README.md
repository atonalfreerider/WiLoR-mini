# WiLoR-mini

implementation of:  
https://github.com/warmshao/WiLoR-mini

# INSTALL  
`conda env create -f environment.yml`

for python 3.5+ compat:
change chumpy/ch.py:

```
def _depends_on(func):
    want_out = 'out' in inspect.getargspec(func).args
```
to 
```
def _depends_on(func):
        want_out = 'out' in inspect.getfullargspec(func).args
```

# RUN  
`python demo.py <path-to-vid>`

output 3d hand poses to hand_poses.json