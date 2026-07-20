# Retinal Dither Names

Recommendation: replace the spoken/written `microsaccade` token in this
world-model context with **Retinal Dither**.

Formal name: **Retinal Mosaic Dither**  
Slug: `retinal_dither`  
Event unit: `dither pulse`

The reason is not just phonetic hygiene, though that matters. `microsaccade`
is easy to mishear as cicade/syccade/cicada-like tokens. More importantly, the
useful substrate here is the retina-scale activation pattern: small image
displacements over a receptor mosaic that renew local contrast, interrupt
adaptation, and can create synchrony structure in retinal ganglion cells.

## Three Interpretations

| Interpretation | Slug | Role | Gay.jl color | Meaning |
|---|---|---:|---|---|
| Retinal Dither | `retinal_dither` | +1 play | `#AFFB3E` | active sampling over a receptor mosaic |
| Mosaic Refresh | `mosaic_refresh` | 0 witness | `#0C068C` | anti-fade renewal that keeps a stable percept live |
| Ganglion Sync | `ganglion_sync` | -1 coplay | `#F5B6F6` | retinal ganglion-cell synchrony as constraint-return code |

The colors were selected with Gay.jl from 900 deterministic candidates using
CIEDE2000 separation. Pairwise distances:

- `#AFFB3E` vs `#0C068C`: `113.63596451748482`
- `#AFFB3E` vs `#F5B6F6`: `72.45014477675956`
- `#0C068C` vs `#F5B6F6`: `67.25885431192907`

## Color Chains

`retinal_dither`, base `#AFFB3E`, chain seed `4708092008076594128`:

```text
#9C5E5C #A5E9B0 #993FC2 #16A66D #99B4FE #A322B5 #8AA2F8
```

`mosaic_refresh`, base `#0C068C`, chain seed `855570893831936446`:

```text
#EF6BB7 #6F0537 #A176D6 #F2CDE7 #C26315 #B91F5F #913DD2
```

`ganglion_sync`, base `#F5B6F6`, chain seed `13499727420772396991`:

```text
#02C534 #B8D4F6 #BE8505 #224614 #D8B048 #897778 #9EB079
```

## Secondary Candidates

- **Foveal Dither** / `foveal_dither`: best if the intended scope is fixation
  and high-acuity foveal sampling.
- **Cone Mosaic Dither** / `cone_mosaic_dither`: more anatomically explicit,
  but longer.
- **Retinotopic Refresh** / `retinotopic_refresh`: good for map-level substrate
  updates.
- **Edge Volley** / `edge_volley`: good for the ganglion synchrony reading, not
  broad enough as the canonical name.
- **Fixation Weave** / `fixation_weave`: good prose for cybernetic interaction,
  less directly retinal.
- **Receptor Resampling** / `receptor_resampling`: precise, but too technical
  for the primary recallable.

## Recallable / Tileable Policy

Use lowercase ASCII slugs. Put substrate first and operation second:
`retinal_dither`, `mosaic_refresh`, `ganglion_sync`.

Avoid `sacc`, `cic`, and `syc` syllables in the canonical token. Let the
productive collisions be with dither, mosaic, refresh, synchrony, coincidence,
and active sampling instead.

Tile URIs:

```text
gay://retina/retinal_dither
gay://retina/mosaic_refresh
gay://retina/ganglion_sync
tile://retinal_dither/receptor_mosaic
tile://mosaic_refresh/adaptation_gate
tile://ganglion_sync/edge_volley
```

The machine-readable version is stored at:

```text
artifacts/retinal_dither_name_world.json
```
