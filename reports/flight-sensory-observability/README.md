# Flight sensory observability

The full 6837-entry encoder produces identical requested-rate vectors for every tested positive/negative native angular-velocity pair at fixed recorded airborne posture. The production current mapper cannot recover a distinction absent from those rates. This is a static native-state capability test, not a neural or flight simulation.

| Probe | Changed inputs | Maximum rate difference (Hz) |
|---|---:|---:|
| roll velocity + versus - 1 rad/s | 0 | 0.000 |
| pitch velocity + versus - 1 rad/s | 0 | 0.000 |
| yaw velocity + versus - 1 rad/s | 0 | 0.000 |
| roll velocity + versus - 10 rad/s | 0 | 0.000 |
| pitch velocity + versus - 10 rad/s | 0 | 0.000 |
| yaw velocity + versus - 10 rad/s | 0 | 0.000 |
| roll velocity + versus - 25 rad/s | 0 | 0.000 |
| pitch velocity + versus - 25 rad/s | 0 | 0.000 |
| yaw velocity + versus - 25 rad/s | 0 | 0.000 |
| roll velocity + versus - 50 rad/s | 0 | 0.000 |
| pitch velocity + versus - 50 rad/s | 0 | 0.000 |
| yaw velocity + versus - 50 rad/s | 0 | 0.000 |
| roll versus pitch magnitude10 | 0 | 0.000 |
| roll versus yaw magnitude10 | 50 | 36.077 |
| mixed-axis full sign reversal | 0 | 0.000 |
| world linear x velocity + versus - 2 cm/s | 0 | 0.000 |
| world linear y velocity + versus - 2 cm/s | 0 | 0.000 |
| world linear z velocity + versus - 2 cm/s | 0 | 0.000 |
| T1-left tibia velocity + versus - 10 rad/s | 0 | 0.000 |
| wing phase field only (downstream dependency probe) | 0 | 0.000 |
| wing frequency field only (downstream dependency probe) | 0 | 0.000 |
| native signed wing joint angles field only (downstream dependency probe) | 0 | 0.000 |
| signed reported pitch/bank field only (downstream dependency probe) | 0 | 0.000 |
| reported altitude field only (downstream dependency probe) | 0 | 0.000 |
| left-only versus right-only wing gate, same rotation | 121 | 45.000 |

Detailed native feedback, organ/side gates, saturation measurements, anatomical grouping fields and source hashes are in result.json. Angular sign is discarded; organ/side identity is retained. Magnitude saturation can additionally erase changes in rotation strength. Future motion, contacts or odor sampling can still change sensory input and are not tested by static sign pairs.
