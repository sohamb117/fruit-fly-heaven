# Recorded wing-contact geometry audit

Read-only classification of selected recorded wing-heightfield contacts; original captured physics scene is compiled only for static rays. No replay, controller, collision or UI changes.

Both contacts at first positive wing impact, maximum wing normal force, and positive wing contacts at peak angular speed and peak fluid root force. This is not an impulse-weighted census of the complete trajectory.

| Event | Time (s) | Normal force | Contact surface height | Rendered top | Nearest visible surface distance | Classification |
|---|---:|---:|---:|---:|---:|---|
| first_impact_1 | 0.11270 | 0.50970 | 16.61614 | 16.43759 | 0.17247 | near_visible_fruit_surface_with_approximation_gap |
| first_impact_2 | 0.11270 | 13.34220 | 16.61943 | 16.45197 | 0.16176 | near_visible_fruit_surface_with_approximation_gap |
| maximum_wing_normal_force | 0.32560 | 350.20108 | 3.75069 | 3.75401 | 0.00326 | visible_bowl_surface |
| angularSpeed_peak_contact_1 | 0.33780 | 102.69527 | 9.52496 | 14.04936 | 0.70627 | invisible_fill_below_visible_fruit |
| angularSpeed_peak_contact_2 | 0.33780 | 65.83958 | 9.13240 | 2.69903 | 1.34523 | heightfield_footprint_cliff_in_visible_air |
| angularSpeed_peak_contact_3 | 0.33780 | 65.46347 | 9.13247 | 2.69903 | 1.34518 | heightfield_footprint_cliff_in_visible_air |
| fluidRootForce_peak_contact_1 | 0.33725 | 8.97022 | 10.44842 | 2.69092 | 0.66776 | heightfield_footprint_cliff_in_visible_air |
| fluidRootForce_peak_contact_2 | 0.33725 | 46.47158 | 10.64411 | 13.06583 | 0.59268 | invisible_fill_below_visible_fruit |
| fluidRootForce_peak_contact_3 | 0.33725 | 46.53351 | 10.64415 | 13.06583 | 0.59267 | invisible_fill_below_visible_fruit |
| fluidRootForce_peak_contact_4 | 0.33725 | 40.40737 | 9.95864 | 14.96105 | 0.27959 | invisible_fill_below_visible_fruit |
| fluidRootForce_peak_contact_5 | 0.33725 | 40.42263 | 9.95872 | 14.96105 | 0.27957 | invisible_fill_below_visible_fruit |
| fluidRootForce_peak_contact_6 | 0.33725 | 65.62158 | 10.17760 | 14.86627 | 0.24745 | invisible_fill_below_visible_fruit |
| fluidRootForce_peak_contact_7 | 0.33725 | 23.97623 | 9.80668 | 13.06583 | 0.80942 | invisible_fill_below_visible_fruit |
| fluidRootForce_peak_contact_8 | 0.33725 | 23.98377 | 9.80669 | 13.06583 | 0.80941 | invisible_fill_below_visible_fruit |
| fluidRootForce_peak_contact_9 | 0.33725 | 21.84782 | 9.79154 | 2.72126 | 0.98306 | heightfield_footprint_cliff_in_visible_air |
| fluidRootForce_peak_contact_10 | 0.33725 | 46.84110 | 10.65844 | 13.06583 | 0.58898 | invisible_fill_below_visible_fruit |
| fluidRootForce_peak_contact_11 | 0.33725 | 46.78104 | 10.65840 | 13.06583 | 0.58899 | invisible_fill_below_visible_fruit |

Heights and distances in the table are scene millimeters; force is g cm/s². Surface positions reconstruct the heightfield-side point from the recorded contact midpoint and penetration distance.

The first wing impact occurs after the first overturn; this audit does not identify the cause of that overturn.

Selected contacts include near-top banana contacts, actual bowl contact, and later invisible heightfield fill/cliff contacts.

An almost horizontal first-contact normal aligns with an internal heightfield grid plane, although neighboring height samples all belong to the same banana. This is a separate prism-seam concern, not a visible fruit silhouette.

The largest normal-force contact lies on the real bowl. Large amplification from a real collision is not by itself evidence of a collision bug.

The recorded behavior still fails: the fly overturns before these impacts. No behavioral success or correction has been demonstrated by this geometry audit.

In a separate matched replay, compare surface-consistent fruit collision volumes while retaining legitimate wing-bowl contacts and explicitly checking internal heightfield seam contacts. Independently investigate wing actuation and fluid coupling before first impact. A terrain correction alone is not established as a cure for the initial overturn.

Scene contact midpoints alone cannot determine impulse causality. The independent matched mechanics replay is the source for intervention effects.

Ray and closest-triangle checks use original main fruit surfaces and corrected original bowl winding. Decorative spots, mold and stems are excluded from nearest-solid distances; they do not explain the selected large gaps.

Banana TubeGeometry has open end rings, so vertical intervals describe the rendered cross-section rather than asserting a closed global volume. Selected under-fruit cases are away from endpoints.

Some contact normals can reflect edges/features. Internal-grid alignment is evidence of a collision representation artifact, not an assertion that changing terrain alone cures the unstable wing drive.
