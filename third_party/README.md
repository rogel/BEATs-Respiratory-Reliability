# Third-party research code used by Gate 9A

`beats/` contains the three runtime files from Microsoft UNILM BEATs at commit
`833df7e7832e5064a281131ee64a481afa8e5b95`. The files remain under Microsoft's
MIT license, copied alongside them as `beats/LICENSE`.

The PANNs CNN6 transfer wrapper is implemented locally because only a small,
weight-compatible subset of the original monolithic `models.py` is required.
Its source version and public checkpoint checksum are frozen in
`artifacts/gate9a_pretrained_ceiling_freeze.json`.
