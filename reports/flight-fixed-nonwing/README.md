# Fixed nonwing mechanics control

All four one-second native runs are retained in result.json. Inputs are identical; the root is free after release. This is not training or a takeoff/landing validation.

The fixed plant removes nonwing joint coordinates at the calibration pose; the dynamic plant retains them under neutral actuator commands. Warm cases have a 100ms restrained startup, with independently recorded initial wing states. No pose resets occur after release.

See the model/source hashes, neutral-geometry invariants, removed joint/armature list, whole-body COM fluid moments, and 2ms samples in result.json.
