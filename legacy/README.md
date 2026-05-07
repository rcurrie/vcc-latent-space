# Legacy

MNIST proxy work that validated the LeWM-inspired architecture (SIGReg + AdaLN + joint training) before pivoting to real scRNA-seq data on the VCC 2025 benchmark. Preserved for reference.

See `../plan.md` for the active roadmap.

- `lewm_scrna.py` — final proxy implementation. Phase 1 (homeostatic, SIGReg) + Phase 2 (perturbation prediction, AdaLN). 3/4 perts passed DR>2.
- `geo_jepa_simple.py` — earlier MCR² + frozen-encoder version. 0/4 perts passed.
- `geo_jepa_mnist.py` — gene-set attention prototype.
- `build_proxy_dataset.py` — MNIST → simulated scRNA-seq generator.
- `u-ctrl-*.py` — unsupervised manifold discovery experiments (predates Geo-JEPA).
- `GEO_JEPA_PLAN.md`, `GEO_JEPA_SESSION_BRIEFING.md` — pre-pivot planning docs.
