# Fruit collision solid validation

Original 60x12 banana tube rings as convex slices; original 12x8 sphere mesh for apples/tips; bowl-only heightfield.

All 17 recorded problem locations were tested using a 0.01 mm radius native sphere. Sixteen locations in visible air now have no contacts; the genuine bowl location retains contact. 65 additional native probes retain legitimate contacts on banana interiors, tips, apples and bowl, including fruit undersides.

47 native ray comparisons against original rendered solid meshes pass; maximum height discrepancy 0.004595263220270596 mm. Native body and joint IDs remain unchanged (68 bodies, 51 joints).

This validates static geometry only. Closed-loop behavior and performance require separate checks. Adjacent banana slices form a compound convex approximation; end caps close the original open tube rings. Decorative spots, mold and stems are not load-bearing collision solids. Bowl heightfield retains its existing tessellation; this helper does not correct unstable wing actuation.
