from dataclasses import dataclass

from jaxtyping import Float
from torch import Tensor


@dataclass
# questo oggetto è relativo alla "nuvola" gaussiana di una SINGOLA SCENA
class Gaussians:
    # qui "batch" è fissato ad 1, ed è messo solo per far tornare la dimensionalità. 
    # quindi ad esempio, se abbiamo una scena per cui sono state generate 150000 (gaussian) gaussiane, questo oggetto avrà: 
    # means: [1, 150000, 3]
    # features: [1, 150000, 256]
    # ...
    means: Float[Tensor, "batch gaussian dim"]  
    covariances: Float[Tensor, "batch gaussian dim dim"]
    harmonics: Float[Tensor, "batch gaussian 3 d_sh"]
    opacities: Float[Tensor, "batch gaussian"]
    features: Float[Tensor, "batch gaussian dim"]
