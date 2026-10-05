# Banana endpoint surface correction

Only createHabitat banana endpoint footprint changed. Original visual fruit geometry and the active observer are unchanged.

Frame150 root projects onto bare bowl beside an interior banana segment, not an endpoint cap. Ground contact points were not recorded, so this correction is not asserted to explain its supported posture.

Frame 150 scene coordinates: (-1.1632325616512753, 12.974724140286762); native floor before/after 2.1281989182653938, rendered floor 2.1319574204937766.

Retain raw nearest-segment projection. A projection beyond the first/last endpoint uses the existing visible radius-2.5 tip sphere instead of a full-radius rounded tube cap. Other nearest interior segments retain the tube radius.

16 isolated native sphere probes contacted the former phantom cap before the fix, had zero contacts at that old height after the fix, and contacted the actual lower bowl after the fix. Maximum corrected native/render height error across bare-bowl removed-cap controls: 0.008164576564976045 scene units. Spawn, apples, inner tube points and frame 150 retain exactly the same analytic surfaces.

A separate 9702-point mesh-ray grid over all six endpoints found 5059 changed positions. The largest visible height above the corrected analytic surface was 0.009514031934306999 scene units; the correction does not discard visible tube geometry. Endpoint-plane roundoff is covered by a focused regression test.

The native 257-point heightfield still approximates abrupt boundaries over one grid cell (0.515625 scene units). Controls adjacent to another fruit footprint are reported separately from bare-bowl contact tests. Existing tube interpolation and low-poly sphere differences remain; no under-fruit voids or overhangs can be represented by a heightfield. No motor or behavior parameters changed and no simulation reload was performed.

2 other controls lie in grid cells crossing the adjacent apple footprint: their native/render height error remains as large as 6.0155970078992205 scene units. This is a separate unresolved heightfield boundary artifact, and those controls are not included in the bare-bowl accuracy claim.
