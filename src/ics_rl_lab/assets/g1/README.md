# Unitree G1 (29 DoF)

`g1.xml` and `meshes/` are copied unchanged (apart from `meshdir="assets"` -> `"meshes"`) from
mjlab 1.2.0's asset zoo (`mjlab/asset_zoo/robots/unitree_g1/xmls/`), which in turn takes the
model from MuJoCo Menagerie's `unitree_g1` (BSD-3-Clause, Unitree Robotics). Check the upstream
licenses before redistributing.

Root body `pelvis`; each foot is seven capsules (`{left,right}_foot[1-7]_collision`).
