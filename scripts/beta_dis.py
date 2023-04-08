# %%
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
import os
import sys


key = random.PRNGKey(0)
x = random.normal(key, (10,))
print(x)



if __name__ == "__main__":
    print("Hello World")
# %%
