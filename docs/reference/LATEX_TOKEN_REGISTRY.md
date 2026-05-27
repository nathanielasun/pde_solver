# LaTeX Token Registry

This document is **generated** from the authoritative C++ registry. Do not edit by hand; regenerate with:

```bash
python3 tools/generate_latex_token_docs.py
```

Sources: [`include/latex_patterns.h`](../include/latex_patterns.h), [`include/latex_token_registry.h`](../include/latex_token_registry.h), [`src/latex_token_registry.cpp`](../src/latex_token_registry.cpp).

## Normalization rules

| From | To |
| --- | --- |
| `\left` | `` |
| `\right` | `` |
| `\cdot` | `*` |
| `\,` | `` |
| `\;` | `` |
| `\:` | `` |
| `\!` | `` |
| `\quad` | `` |
| `\qquad` | `` |
| `$` | `` |
| `\theta` | `theta` |
| `\phi` | `phi` |
| `^{2}` | `^2` |
| `_{xx}` | `_xx` |
| `_{yy}` | `_yy` |
| `_{zz}` | `_zz` |
| `_{x}` | `_x` |
| `_{y}` | `_y` |
| `_{t}` | `_t` |
| `_{r}` | `_r` |
| `_{theta}` | `_theta` |
| `_{phi}` | `_phi` |

## Conservation rewrites (parse-time)

| From | To | Note |
| --- | --- | --- |
| `d/dx(0.5*u^2)` | `u*u_x` | Burgers flux divergence |
| `d/dx(0.5*u*u)` | `u*u_x` | Burgers flux divergence |
| `\frac{d}{dx}(0.5*u^2)` | `u*u_x` | Burgers flux divergence |

## PDE pattern tokens by category

### derivative_d2t

| Pattern | Maps to | Display |
| --- | --- | --- |
| `\frac{\partial^2u}{\partialt^2}` | `PDECoefficients.utt` | ∂²u/∂t² |
| `\frac{\partial^2u}{\partialt\partialt}` | `PDECoefficients.utt` | ∂²u/∂t² |
| `\partial_{tt}u` | `PDECoefficients.utt` | ∂²u/∂t² |
| `\partial_t\partial_tu` | `PDECoefficients.utt` | ∂²u/∂t² |
| `u_{tt}` | `PDECoefficients.utt` | ∂²u/∂t² |
| `\partial^2u/\partialt^2` | `PDECoefficients.utt` | ∂²u/∂t² |
| `\partial^2u/\partialt\partialt` | `PDECoefficients.utt` | ∂²u/∂t² |
| `\frac{d^2u}{dt^2}` | `PDECoefficients.utt` | ∂²u/∂t² |
| `\frac{d^2u}{dtdt}` | `PDECoefficients.utt` | ∂²u/∂t² |
| `d^2u/dt^2` | `PDECoefficients.utt` | ∂²u/∂t² |
| `d^2u/dtdt` | `PDECoefficients.utt` | ∂²u/∂t² |
| `\ddot{u}` | `PDECoefficients.utt` | ∂²u/∂t² |
| `\ddotu` | `PDECoefficients.utt` | ∂²u/∂t² |

### derivative_d2x

| Pattern | Maps to | Display |
| --- | --- | --- |
| `\frac{\partial^2u}{\partialx^2}` | `PDECoefficients.a` | ∂²u/∂x² |
| `\frac{\partial^2u}{\partialx\partialx}` | `PDECoefficients.a` | ∂²u/∂x² |
| `\partial_{xx}u` | `PDECoefficients.a` | ∂²u/∂x² |
| `\partial_x\partial_xu` | `PDECoefficients.a` | ∂²u/∂x² |
| `u_{xx}` | `PDECoefficients.a` | ∂²u/∂x² |
| `u_xx` | `PDECoefficients.a` | ∂²u/∂x² |
| `\partial^2u/\partialx^2` | `PDECoefficients.a` | ∂²u/∂x² |
| `\partial^2u/\partialx\partialx` | `PDECoefficients.a` | ∂²u/∂x² |
| `\frac{d^2u}{dx^2}` | `PDECoefficients.a` | ∂²u/∂x² |
| `\frac{d^2u}{dxdx}` | `PDECoefficients.a` | ∂²u/∂x² |
| `d^2u/dx^2` | `PDECoefficients.a` | ∂²u/∂x² |
| `d^2u/dxdx` | `PDECoefficients.a` | ∂²u/∂x² |
| `\frac{\partial^2u}{\partialr^2}` | `PDECoefficients.a` | ∂²u/∂x² |
| `\frac{\partial^2u}{\partialr\partialr}` | `PDECoefficients.a` | ∂²u/∂x² |
| `\partial_{rr}u` | `PDECoefficients.a` | ∂²u/∂x² |
| `\partial_r\partial_ru` | `PDECoefficients.a` | ∂²u/∂x² |
| `u_{rr}` | `PDECoefficients.a` | ∂²u/∂x² |
| `\partial^2u/\partialr^2` | `PDECoefficients.a` | ∂²u/∂x² |
| `\partial^2u/\partialr\partialr` | `PDECoefficients.a` | ∂²u/∂x² |
| `\frac{d^2u}{dr^2}` | `PDECoefficients.a` | ∂²u/∂x² |
| `\frac{d^2u}{drdr}` | `PDECoefficients.a` | ∂²u/∂x² |
| `d^2u/dr^2` | `PDECoefficients.a` | ∂²u/∂x² |
| `d^2u/drdr` | `PDECoefficients.a` | ∂²u/∂x² |

### derivative_d2y

| Pattern | Maps to | Display |
| --- | --- | --- |
| `\frac{\partial^2u}{\partialy^2}` | `PDECoefficients.b` | ∂²u/∂y² |
| `\frac{\partial^2u}{\partialy\partialy}` | `PDECoefficients.b` | ∂²u/∂y² |
| `\partial_{yy}u` | `PDECoefficients.b` | ∂²u/∂y² |
| `\partial_y\partial_yu` | `PDECoefficients.b` | ∂²u/∂y² |
| `u_{yy}` | `PDECoefficients.b` | ∂²u/∂y² |
| `\partial^2u/\partialy^2` | `PDECoefficients.b` | ∂²u/∂y² |
| `\partial^2u/\partialy\partialy` | `PDECoefficients.b` | ∂²u/∂y² |
| `\frac{d^2u}{dy^2}` | `PDECoefficients.b` | ∂²u/∂y² |
| `\frac{d^2u}{dydy}` | `PDECoefficients.b` | ∂²u/∂y² |
| `d^2u/dy^2` | `PDECoefficients.b` | ∂²u/∂y² |
| `d^2u/dydy` | `PDECoefficients.b` | ∂²u/∂y² |
| `\frac{\partial^2u}{\partialtheta^2}` | `PDECoefficients.b` | ∂²u/∂y² |
| `\frac{\partial^2u}{\partialtheta\partialtheta}` | `PDECoefficients.b` | ∂²u/∂y² |
| `\partial_{thetatheta}u` | `PDECoefficients.b` | ∂²u/∂y² |
| `\partial_theta\partial_thetau` | `PDECoefficients.b` | ∂²u/∂y² |
| `u_{thetatheta}` | `PDECoefficients.b` | ∂²u/∂y² |
| `\partial^2u/\partialtheta^2` | `PDECoefficients.b` | ∂²u/∂y² |
| `\partial^2u/\partialtheta\partialtheta` | `PDECoefficients.b` | ∂²u/∂y² |
| `\frac{d^2u}{dtheta^2}` | `PDECoefficients.b` | ∂²u/∂y² |
| `\frac{d^2u}{dthetadtheta}` | `PDECoefficients.b` | ∂²u/∂y² |
| `d^2u/dtheta^2` | `PDECoefficients.b` | ∂²u/∂y² |
| `d^2u/dthetadtheta` | `PDECoefficients.b` | ∂²u/∂y² |

### derivative_d2z

| Pattern | Maps to | Display |
| --- | --- | --- |
| `\frac{\partial^2u}{\partialz^2}` | `PDECoefficients.az` | ∂²u/∂z² |
| `\frac{\partial^2u}{\partialz\partialz}` | `PDECoefficients.az` | ∂²u/∂z² |
| `\partial_{zz}u` | `PDECoefficients.az` | ∂²u/∂z² |
| `\partial_z\partial_zu` | `PDECoefficients.az` | ∂²u/∂z² |
| `u_{zz}` | `PDECoefficients.az` | ∂²u/∂z² |
| `\partial^2u/\partialz^2` | `PDECoefficients.az` | ∂²u/∂z² |
| `\partial^2u/\partialz\partialz` | `PDECoefficients.az` | ∂²u/∂z² |
| `\frac{d^2u}{dz^2}` | `PDECoefficients.az` | ∂²u/∂z² |
| `\frac{d^2u}{dzdz}` | `PDECoefficients.az` | ∂²u/∂z² |
| `d^2u/dz^2` | `PDECoefficients.az` | ∂²u/∂z² |
| `d^2u/dzdz` | `PDECoefficients.az` | ∂²u/∂z² |
| `\frac{\partial^2u}{\partialphi^2}` | `PDECoefficients.az` | ∂²u/∂z² |
| `\frac{\partial^2u}{\partialphi\partialphi}` | `PDECoefficients.az` | ∂²u/∂z² |
| `\partial_{phiphi}u` | `PDECoefficients.az` | ∂²u/∂z² |
| `\partial_phi\partial_phiu` | `PDECoefficients.az` | ∂²u/∂z² |
| `u_{phiphi}` | `PDECoefficients.az` | ∂²u/∂z² |
| `\partial^2u/\partialphi^2` | `PDECoefficients.az` | ∂²u/∂z² |
| `\partial^2u/\partialphi\partialphi` | `PDECoefficients.az` | ∂²u/∂z² |
| `\frac{d^2u}{dphi^2}` | `PDECoefficients.az` | ∂²u/∂z² |
| `\frac{d^2u}{dphidphi}` | `PDECoefficients.az` | ∂²u/∂z² |
| `d^2u/dphi^2` | `PDECoefficients.az` | ∂²u/∂z² |
| `d^2u/dphidphi` | `PDECoefficients.az` | ∂²u/∂z² |

### derivative_d3x

| Pattern | Maps to | Display |
| --- | --- | --- |
| `\frac{\partial^3u}{\partialx^3}` | `PDECoefficients.a3` | ∂³u/∂x³ |
| `\frac{\partial^3u}{\partialx\partialx\partialx}` | `PDECoefficients.a3` | ∂³u/∂x³ |
| `\partial_{xxx}u` | `PDECoefficients.a3` | ∂³u/∂x³ |
| `\partial_x\partial_x\partial_xu` | `PDECoefficients.a3` | ∂³u/∂x³ |
| `u_{xxx}` | `PDECoefficients.a3` | ∂³u/∂x³ |
| `\partial^3u/\partialx^3` | `PDECoefficients.a3` | ∂³u/∂x³ |
| `\partial^3u/\partialx\partialx\partialx` | `PDECoefficients.a3` | ∂³u/∂x³ |
| `\frac{d^3u}{dx^3}` | `PDECoefficients.a3` | ∂³u/∂x³ |
| `\frac{d^3u}{dxdxdx}` | `PDECoefficients.a3` | ∂³u/∂x³ |
| `d^3u/dx^3` | `PDECoefficients.a3` | ∂³u/∂x³ |
| `d^3u/dxdxdx` | `PDECoefficients.a3` | ∂³u/∂x³ |
| `\frac{\partial^3u}{\partialr^3}` | `PDECoefficients.a3` | ∂³u/∂x³ |
| `\partial_{rrr}u` | `PDECoefficients.a3` | ∂³u/∂x³ |
| `u_{rrr}` | `PDECoefficients.a3` | ∂³u/∂x³ |

### derivative_d3y

| Pattern | Maps to | Display |
| --- | --- | --- |
| `\frac{\partial^3u}{\partialy^3}` | `PDECoefficients.b3` | ∂³u/∂y³ |
| `\frac{\partial^3u}{\partialy\partialy\partialy}` | `PDECoefficients.b3` | ∂³u/∂y³ |
| `\partial_{yyy}u` | `PDECoefficients.b3` | ∂³u/∂y³ |
| `\partial_y\partial_y\partial_yu` | `PDECoefficients.b3` | ∂³u/∂y³ |
| `u_{yyy}` | `PDECoefficients.b3` | ∂³u/∂y³ |
| `\partial^3u/\partialy^3` | `PDECoefficients.b3` | ∂³u/∂y³ |
| `\partial^3u/\partialy\partialy\partialy` | `PDECoefficients.b3` | ∂³u/∂y³ |
| `\frac{d^3u}{dy^3}` | `PDECoefficients.b3` | ∂³u/∂y³ |
| `\frac{d^3u}{dydydy}` | `PDECoefficients.b3` | ∂³u/∂y³ |
| `d^3u/dy^3` | `PDECoefficients.b3` | ∂³u/∂y³ |
| `d^3u/dydydy` | `PDECoefficients.b3` | ∂³u/∂y³ |
| `\frac{\partial^3u}{\partialtheta^3}` | `PDECoefficients.b3` | ∂³u/∂y³ |
| `\partial_{thetathetatheta}u` | `PDECoefficients.b3` | ∂³u/∂y³ |
| `u_{thetathetatheta}` | `PDECoefficients.b3` | ∂³u/∂y³ |

### derivative_d3z

| Pattern | Maps to | Display |
| --- | --- | --- |
| `\frac{\partial^3u}{\partialz^3}` | `PDECoefficients.az3` | ∂³u/∂z³ |
| `\frac{\partial^3u}{\partialz\partialz\partialz}` | `PDECoefficients.az3` | ∂³u/∂z³ |
| `\partial_{zzz}u` | `PDECoefficients.az3` | ∂³u/∂z³ |
| `\partial_z\partial_z\partial_zu` | `PDECoefficients.az3` | ∂³u/∂z³ |
| `u_{zzz}` | `PDECoefficients.az3` | ∂³u/∂z³ |
| `\partial^3u/\partialz^3` | `PDECoefficients.az3` | ∂³u/∂z³ |
| `\partial^3u/\partialz\partialz\partialz` | `PDECoefficients.az3` | ∂³u/∂z³ |
| `\frac{d^3u}{dz^3}` | `PDECoefficients.az3` | ∂³u/∂z³ |
| `\frac{d^3u}{dzdzdz}` | `PDECoefficients.az3` | ∂³u/∂z³ |
| `d^3u/dz^3` | `PDECoefficients.az3` | ∂³u/∂z³ |
| `d^3u/dzdzdz` | `PDECoefficients.az3` | ∂³u/∂z³ |
| `\frac{\partial^3u}{\partialphi^3}` | `PDECoefficients.az3` | ∂³u/∂z³ |
| `\partial_{phiphiphi}u` | `PDECoefficients.az3` | ∂³u/∂z³ |
| `u_{phiphiphi}` | `PDECoefficients.az3` | ∂³u/∂z³ |

### derivative_d4x

| Pattern | Maps to | Display |
| --- | --- | --- |
| `\frac{\partial^4u}{\partialx^4}` | `PDECoefficients.a4` | ∂⁴u/∂x⁴ |
| `\frac{\partial^4u}{\partialx\partialx\partialx\partialx}` | `PDECoefficients.a4` | ∂⁴u/∂x⁴ |
| `\partial_{xxxx}u` | `PDECoefficients.a4` | ∂⁴u/∂x⁴ |
| `\partial_x\partial_x\partial_x\partial_xu` | `PDECoefficients.a4` | ∂⁴u/∂x⁴ |
| `u_{xxxx}` | `PDECoefficients.a4` | ∂⁴u/∂x⁴ |
| `\partial^4u/\partialx^4` | `PDECoefficients.a4` | ∂⁴u/∂x⁴ |
| `\partial^4u/\partialx\partialx\partialx\partialx` | `PDECoefficients.a4` | ∂⁴u/∂x⁴ |
| `\frac{d^4u}{dx^4}` | `PDECoefficients.a4` | ∂⁴u/∂x⁴ |
| `\frac{d^4u}{dxdxdxdx}` | `PDECoefficients.a4` | ∂⁴u/∂x⁴ |
| `d^4u/dx^4` | `PDECoefficients.a4` | ∂⁴u/∂x⁴ |
| `d^4u/dxdxdxdx` | `PDECoefficients.a4` | ∂⁴u/∂x⁴ |
| `\frac{\partial^4u}{\partialr^4}` | `PDECoefficients.a4` | ∂⁴u/∂x⁴ |
| `\partial_{rrrr}u` | `PDECoefficients.a4` | ∂⁴u/∂x⁴ |
| `u_{rrrr}` | `PDECoefficients.a4` | ∂⁴u/∂x⁴ |

### derivative_d4y

| Pattern | Maps to | Display |
| --- | --- | --- |
| `\frac{\partial^4u}{\partialy^4}` | `PDECoefficients.b4` | ∂⁴u/∂y⁴ |
| `\frac{\partial^4u}{\partialy\partialy\partialy\partialy}` | `PDECoefficients.b4` | ∂⁴u/∂y⁴ |
| `\partial_{yyyy}u` | `PDECoefficients.b4` | ∂⁴u/∂y⁴ |
| `\partial_y\partial_y\partial_y\partial_yu` | `PDECoefficients.b4` | ∂⁴u/∂y⁴ |
| `u_{yyyy}` | `PDECoefficients.b4` | ∂⁴u/∂y⁴ |
| `\partial^4u/\partialy^4` | `PDECoefficients.b4` | ∂⁴u/∂y⁴ |
| `\partial^4u/\partialy\partialy\partialy\partialy` | `PDECoefficients.b4` | ∂⁴u/∂y⁴ |
| `\frac{d^4u}{dy^4}` | `PDECoefficients.b4` | ∂⁴u/∂y⁴ |
| `\frac{d^4u}{dydydydy}` | `PDECoefficients.b4` | ∂⁴u/∂y⁴ |
| `d^4u/dy^4` | `PDECoefficients.b4` | ∂⁴u/∂y⁴ |
| `d^4u/dydydydy` | `PDECoefficients.b4` | ∂⁴u/∂y⁴ |
| `\frac{\partial^4u}{\partialtheta^4}` | `PDECoefficients.b4` | ∂⁴u/∂y⁴ |
| `\partial_{thetathetathetatheta}u` | `PDECoefficients.b4` | ∂⁴u/∂y⁴ |
| `u_{thetathetathetatheta}` | `PDECoefficients.b4` | ∂⁴u/∂y⁴ |

### derivative_d4z

| Pattern | Maps to | Display |
| --- | --- | --- |
| `\frac{\partial^4u}{\partialz^4}` | `PDECoefficients.az4` | ∂⁴u/∂z⁴ |
| `\frac{\partial^4u}{\partialz\partialz\partialz\partialz}` | `PDECoefficients.az4` | ∂⁴u/∂z⁴ |
| `\partial_{zzzz}u` | `PDECoefficients.az4` | ∂⁴u/∂z⁴ |
| `\partial_z\partial_z\partial_z\partial_zu` | `PDECoefficients.az4` | ∂⁴u/∂z⁴ |
| `u_{zzzz}` | `PDECoefficients.az4` | ∂⁴u/∂z⁴ |
| `\partial^4u/\partialz^4` | `PDECoefficients.az4` | ∂⁴u/∂z⁴ |
| `\partial^4u/\partialz\partialz\partialz\partialz` | `PDECoefficients.az4` | ∂⁴u/∂z⁴ |
| `\frac{d^4u}{dz^4}` | `PDECoefficients.az4` | ∂⁴u/∂z⁴ |
| `\frac{d^4u}{dzdzdzdz}` | `PDECoefficients.az4` | ∂⁴u/∂z⁴ |
| `d^4u/dz^4` | `PDECoefficients.az4` | ∂⁴u/∂z⁴ |
| `d^4u/dzdzdzdz` | `PDECoefficients.az4` | ∂⁴u/∂z⁴ |
| `\frac{\partial^4u}{\partialphi^4}` | `PDECoefficients.az4` | ∂⁴u/∂z⁴ |
| `\partial_{phiphiphiphi}u` | `PDECoefficients.az4` | ∂⁴u/∂z⁴ |
| `u_{phiphiphiphi}` | `PDECoefficients.az4` | ∂⁴u/∂z⁴ |

### derivative_dt

| Pattern | Maps to | Display |
| --- | --- | --- |
| `\frac{\partialu}{\partialt}` | `PDECoefficients.ut` | ∂u/∂t |
| `\partial_tu` | `PDECoefficients.ut` | ∂u/∂t |
| `u_t` | `PDECoefficients.ut` | ∂u/∂t |
| `\partialu/\partialt` | `PDECoefficients.ut` | ∂u/∂t |
| `\frac{du}{dt}` | `PDECoefficients.ut` | ∂u/∂t |
| `du/dt` | `PDECoefficients.ut` | ∂u/∂t |
| `\dot{u}` | `PDECoefficients.ut` | ∂u/∂t |
| `\dotu` | `PDECoefficients.ut` | ∂u/∂t |

### derivative_dx

| Pattern | Maps to | Display |
| --- | --- | --- |
| `\frac{\partialu}{\partialx}` | `PDECoefficients.c` | ∂u/∂x |
| `\partial_xu` | `PDECoefficients.c` | ∂u/∂x |
| `u_x` | `PDECoefficients.c` | ∂u/∂x |
| `\partialu/\partialx` | `PDECoefficients.c` | ∂u/∂x |
| `\frac{du}{dx}` | `PDECoefficients.c` | ∂u/∂x |
| `du/dx` | `PDECoefficients.c` | ∂u/∂x |
| `\frac{\partialu}{\partialr}` | `PDECoefficients.c` | ∂u/∂x |
| `\partial_ru` | `PDECoefficients.c` | ∂u/∂x |
| `u_r` | `PDECoefficients.c` | ∂u/∂x |
| `\partialu/\partialr` | `PDECoefficients.c` | ∂u/∂x |
| `\frac{du}{dr}` | `PDECoefficients.c` | ∂u/∂x |
| `du/dr` | `PDECoefficients.c` | ∂u/∂x |

### derivative_dy

| Pattern | Maps to | Display |
| --- | --- | --- |
| `\frac{\partialu}{\partialy}` | `PDECoefficients.d` | ∂u/∂y |
| `\partial_yu` | `PDECoefficients.d` | ∂u/∂y |
| `u_y` | `PDECoefficients.d` | ∂u/∂y |
| `\partialu/\partialy` | `PDECoefficients.d` | ∂u/∂y |
| `\frac{du}{dy}` | `PDECoefficients.d` | ∂u/∂y |
| `du/dy` | `PDECoefficients.d` | ∂u/∂y |
| `\frac{\partialu}{\partialtheta}` | `PDECoefficients.d` | ∂u/∂y |
| `\partial_thetau` | `PDECoefficients.d` | ∂u/∂y |
| `u_theta` | `PDECoefficients.d` | ∂u/∂y |
| `\partialu/\partialtheta` | `PDECoefficients.d` | ∂u/∂y |
| `\frac{du}{dtheta}` | `PDECoefficients.d` | ∂u/∂y |
| `du/dtheta` | `PDECoefficients.d` | ∂u/∂y |

### derivative_dz

| Pattern | Maps to | Display |
| --- | --- | --- |
| `\frac{\partialu}{\partialz}` | `PDECoefficients.dz` | ∂u/∂z |
| `\partial_zu` | `PDECoefficients.dz` | ∂u/∂z |
| `u_z` | `PDECoefficients.dz` | ∂u/∂z |
| `\partialu/\partialz` | `PDECoefficients.dz` | ∂u/∂z |
| `\frac{du}{dz}` | `PDECoefficients.dz` | ∂u/∂z |
| `du/dz` | `PDECoefficients.dz` | ∂u/∂z |
| `\frac{\partialu}{\partialphi}` | `PDECoefficients.dz` | ∂u/∂z |
| `\partial_phiu` | `PDECoefficients.dz` | ∂u/∂z |
| `u_phi` | `PDECoefficients.dz` | ∂u/∂z |
| `\partialu/\partialphi` | `PDECoefficients.dz` | ∂u/∂z |
| `\frac{du}{dphi}` | `PDECoefficients.dz` | ∂u/∂z |
| `du/dphi` | `PDECoefficients.dz` | ∂u/∂z |

### derivative_mixed_xy

| Pattern | Maps to | Display |
| --- | --- | --- |
| `u_{xy}` | `PDECoefficients.ab` | ∂²u/∂x∂y |
| `u_{yx}` | `PDECoefficients.ab` | ∂²u/∂x∂y |
| `\partial_{xy}u` | `PDECoefficients.ab` | ∂²u/∂x∂y |
| `\partial_{yx}u` | `PDECoefficients.ab` | ∂²u/∂x∂y |
| `\partial_x\partial_yu` | `PDECoefficients.ab` | ∂²u/∂x∂y |
| `\partial_y\partial_xu` | `PDECoefficients.ab` | ∂²u/∂x∂y |
| `\frac{\partial^2u}{\partialx\partialy}` | `PDECoefficients.ab` | ∂²u/∂x∂y |
| `\frac{\partial^2u}{\partialy\partialx}` | `PDECoefficients.ab` | ∂²u/∂x∂y |
| `\partial^2u/\partialx\partialy` | `PDECoefficients.ab` | ∂²u/∂x∂y |
| `\partial^2u/\partialy\partialx` | `PDECoefficients.ab` | ∂²u/∂x∂y |
| `\frac{d^2u}{dxdy}` | `PDECoefficients.ab` | ∂²u/∂x∂y |
| `\frac{d^2u}{dydx}` | `PDECoefficients.ab` | ∂²u/∂x∂y |
| `d^2u/dxdy` | `PDECoefficients.ab` | ∂²u/∂x∂y |
| `d^2u/dydx` | `PDECoefficients.ab` | ∂²u/∂x∂y |
| `u_{rtheta}` | `PDECoefficients.ab` | ∂²u/∂x∂y |
| `u_{thetar}` | `PDECoefficients.ab` | ∂²u/∂x∂y |
| `\partial_{rtheta}u` | `PDECoefficients.ab` | ∂²u/∂x∂y |
| `\partial_{thetar}u` | `PDECoefficients.ab` | ∂²u/∂x∂y |
| `\partial_r\partial_thetau` | `PDECoefficients.ab` | ∂²u/∂x∂y |
| `\partial_theta\partial_ru` | `PDECoefficients.ab` | ∂²u/∂x∂y |
| `\frac{\partial^2u}{\partialr\partialtheta}` | `PDECoefficients.ab` | ∂²u/∂x∂y |
| `\frac{\partial^2u}{\partialtheta\partialr}` | `PDECoefficients.ab` | ∂²u/∂x∂y |
| `\partial^2u/\partialr\partialtheta` | `PDECoefficients.ab` | ∂²u/∂x∂y |
| `\partial^2u/\partialtheta\partialr` | `PDECoefficients.ab` | ∂²u/∂x∂y |

### derivative_mixed_xz

| Pattern | Maps to | Display |
| --- | --- | --- |
| `u_{xz}` | `PDECoefficients.ac` | ∂²u/∂x∂z |
| `u_{zx}` | `PDECoefficients.ac` | ∂²u/∂x∂z |
| `\partial_{xz}u` | `PDECoefficients.ac` | ∂²u/∂x∂z |
| `\partial_{zx}u` | `PDECoefficients.ac` | ∂²u/∂x∂z |
| `\partial_x\partial_zu` | `PDECoefficients.ac` | ∂²u/∂x∂z |
| `\partial_z\partial_xu` | `PDECoefficients.ac` | ∂²u/∂x∂z |
| `\frac{\partial^2u}{\partialx\partialz}` | `PDECoefficients.ac` | ∂²u/∂x∂z |
| `\frac{\partial^2u}{\partialz\partialx}` | `PDECoefficients.ac` | ∂²u/∂x∂z |
| `\partial^2u/\partialx\partialz` | `PDECoefficients.ac` | ∂²u/∂x∂z |
| `\partial^2u/\partialz\partialx` | `PDECoefficients.ac` | ∂²u/∂x∂z |
| `\frac{d^2u}{dxdz}` | `PDECoefficients.ac` | ∂²u/∂x∂z |
| `\frac{d^2u}{dzdx}` | `PDECoefficients.ac` | ∂²u/∂x∂z |
| `d^2u/dxdz` | `PDECoefficients.ac` | ∂²u/∂x∂z |
| `d^2u/dzdx` | `PDECoefficients.ac` | ∂²u/∂x∂z |
| `u_{rphi}` | `PDECoefficients.ac` | ∂²u/∂x∂z |
| `u_{phir}` | `PDECoefficients.ac` | ∂²u/∂x∂z |
| `\partial_{rphi}u` | `PDECoefficients.ac` | ∂²u/∂x∂z |
| `\partial_{phir}u` | `PDECoefficients.ac` | ∂²u/∂x∂z |
| `\partial_r\partial_phiu` | `PDECoefficients.ac` | ∂²u/∂x∂z |
| `\partial_phi\partial_ru` | `PDECoefficients.ac` | ∂²u/∂x∂z |
| `\frac{\partial^2u}{\partialr\partialphi}` | `PDECoefficients.ac` | ∂²u/∂x∂z |
| `\frac{\partial^2u}{\partialphi\partialr}` | `PDECoefficients.ac` | ∂²u/∂x∂z |
| `\partial^2u/\partialr\partialphi` | `PDECoefficients.ac` | ∂²u/∂x∂z |
| `\partial^2u/\partialphi\partialr` | `PDECoefficients.ac` | ∂²u/∂x∂z |

### derivative_mixed_yz

| Pattern | Maps to | Display |
| --- | --- | --- |
| `u_{yz}` | `PDECoefficients.bc` | ∂²u/∂y∂z |
| `u_{zy}` | `PDECoefficients.bc` | ∂²u/∂y∂z |
| `\partial_{yz}u` | `PDECoefficients.bc` | ∂²u/∂y∂z |
| `\partial_{zy}u` | `PDECoefficients.bc` | ∂²u/∂y∂z |
| `\partial_y\partial_zu` | `PDECoefficients.bc` | ∂²u/∂y∂z |
| `\partial_z\partial_yu` | `PDECoefficients.bc` | ∂²u/∂y∂z |
| `\frac{\partial^2u}{\partialy\partialz}` | `PDECoefficients.bc` | ∂²u/∂y∂z |
| `\frac{\partial^2u}{\partialz\partialy}` | `PDECoefficients.bc` | ∂²u/∂y∂z |
| `\partial^2u/\partialy\partialz` | `PDECoefficients.bc` | ∂²u/∂y∂z |
| `\partial^2u/\partialz\partialy` | `PDECoefficients.bc` | ∂²u/∂y∂z |
| `\frac{d^2u}{dydz}` | `PDECoefficients.bc` | ∂²u/∂y∂z |
| `\frac{d^2u}{dzdy}` | `PDECoefficients.bc` | ∂²u/∂y∂z |
| `d^2u/dydz` | `PDECoefficients.bc` | ∂²u/∂y∂z |
| `d^2u/dzdy` | `PDECoefficients.bc` | ∂²u/∂y∂z |
| `u_{thetaphi}` | `PDECoefficients.bc` | ∂²u/∂y∂z |
| `u_{phitheta}` | `PDECoefficients.bc` | ∂²u/∂y∂z |
| `\partial_{thetaphi}u` | `PDECoefficients.bc` | ∂²u/∂y∂z |
| `\partial_{phitheta}u` | `PDECoefficients.bc` | ∂²u/∂y∂z |
| `\partial_theta\partial_phiu` | `PDECoefficients.bc` | ∂²u/∂y∂z |
| `\partial_phi\partial_thetau` | `PDECoefficients.bc` | ∂²u/∂y∂z |
| `\frac{\partial^2u}{\partialtheta\partialphi}` | `PDECoefficients.bc` | ∂²u/∂y∂z |
| `\frac{\partial^2u}{\partialphi\partialtheta}` | `PDECoefficients.bc` | ∂²u/∂y∂z |
| `\partial^2u/\partialtheta\partialphi` | `PDECoefficients.bc` | ∂²u/∂y∂z |
| `\partial^2u/\partialphi\partialtheta` | `PDECoefficients.bc` | ∂²u/∂y∂z |

### display_macro

| Pattern | Maps to | Display |
| --- | --- | --- |
| `\frac{a}{b}` | `MicroTeX preview` | fraction |
| `\sin` | `MicroTeX preview` | sine |
| `\cos` | `MicroTeX preview` | cosine |
| `\exp` | `MicroTeX preview` | exponential |
| `\sqrt{x}` | `MicroTeX preview` | square root |
| `\partial` | `MicroTeX preview` | partial derivative |
| `\nabla` | `MicroTeX preview` | nabla |
| `u_{xx}` | `MicroTeX preview` | subscript |
| `x^{2}` | `MicroTeX preview` | superscript |

### integral

| Pattern | Maps to | Display |
| --- | --- | --- |
| `\int u` | `IntegralTerm` | global integral of u |
| `\int_{domain} u` | `IntegralTerm` | domain integral |

### integral_kernel

| Pattern | Maps to | Display |
| --- | --- | --- |
| `\int K(x,y) u` | `IntegralTerm.kernel_latex` | kernel integral |

### laplacian

| Pattern | Maps to | Display |
| --- | --- | --- |
| `\nabla^2u` | `PDECoefficients.a+b(+az)` | ∇²u |
| `\Deltau` | `PDECoefficients.a+b(+az)` | ∇²u |
| `\triangleu` | `PDECoefficients.a+b(+az)` | ∇²u |

### nonlinear_abs

| Pattern | Maps to | Display |
| --- | --- | --- |
| `|u|` | `NonlinearTerm.Abs` |  |

### nonlinear_cos

| Pattern | Maps to | Display |
| --- | --- | --- |
| `\cos(u)` | `NonlinearTerm.Cos` |  |
| `cos(u)` | `NonlinearTerm.Cos` |  |

### nonlinear_derivative

| Pattern | Maps to | Display |
| --- | --- | --- |
| `u*u_x` | `NonlinearDerivativeTerm.UUx` |  |
| `u*u_y` | `NonlinearDerivativeTerm.UUy` |  |
| `u*u_z` | `NonlinearDerivativeTerm.UUz` |  |
| `u_x^2` | `NonlinearDerivativeTerm.UxUx` |  |
| `u_y^2` | `NonlinearDerivativeTerm.UyUy` |  |
| `u_z^2` | `NonlinearDerivativeTerm.UzUz` |  |
| `|\nablau|^2` | `NonlinearDerivativeTerm.GradSquared` |  |
| `u_x^2+u_y^2` | `NonlinearDerivativeTerm.GradSquared` |  |
| `u_x^2+u_y^2+u_z^2` | `NonlinearDerivativeTerm.GradSquared` |  |

### nonlinear_exp

| Pattern | Maps to | Display |
| --- | --- | --- |
| `\exp(u)` | `NonlinearTerm.Exp` |  |
| `exp(u)` | `NonlinearTerm.Exp` |  |

### nonlinear_power

| Pattern | Maps to | Display |
| --- | --- | --- |
| `u^{n}` | `NonlinearTerm.Power` | u^n (n>=2) |

### nonlinear_sin

| Pattern | Maps to | Display |
| --- | --- | --- |
| `\sin(u)` | `NonlinearTerm.Sin` |  |
| `sin(u)` | `NonlinearTerm.Sin` |  |

### state_variable

| Pattern | Maps to | Display |
| --- | --- | --- |
| `u` | `PDECoefficients.e` | u |
