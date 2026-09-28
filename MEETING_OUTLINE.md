# 6-slide meeting outline

1. **Motivation** — structure-based virtual screening needs a score for protein–ligand interactions.
2. **Connection to Hou Lab** — IGN learns protein–ligand interaction patterns from 3D complexes; RTMScore explicitly combines a 3D residue representation, a 2D ligand graph, and residue–atom distance information.
3. **Question** — does coarse pocket geometry add predictive information beyond ligand-only fingerprints?
4. **Method** — PDBBind Core → RDKit ligand features + pocket residue/distance histogram → two matched PyTorch MLPs → 5-fold OOF evaluation.
5. **Results** — show `figs/02_model_comparison.png`, `figs/01_parity.png`, and `figs/04_pocket_heatmap.png`.
6. **Next step** — replace the hand-crafted pocket fingerprint with a true residue–atom graph and compare against IGN/RTMScore on a larger split; move the experiment to Linux/GPU when scaling up.
