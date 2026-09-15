# Additional power smoothing: completed trace comparison

The 20 ms power filter made this development seed descend into contact sooner. It did not cause an initial inversion. Identical release state, initial observation, inputs through 500 ms, first command, fitted weights, and the filter's power-only mask were verified. The body and neural trajectory subsequently diverged in closed loop.

| First 0.3 seconds | Identity | 20 ms power smoothing |
|---|---:|---:|
| Mean left / right power | 0.799964 / 0.796130 | 0.785337 / 0.777205 |
| Mean common power | 0.798047 | 0.781271 |
| Common power SD | 0.011707 | 0.013235 |
| Mean left-minus-right power | 0.003834 | 0.008132 |
| Differential power RMS | 0.007968 | 0.014086 |
| Height at 0.3 s, cm | 2.924938 | 2.495349 |
| Height-derived average vertical speed, cm/s | -1.859744 | -3.291710 |
| Maximum tilt, degrees | 15.854 | 20.686 |
| Root angular-speed RMS, rad/s | 8.231 | 8.674 |
| Maximum root angular speed, rad/s | 18.506 | 18.900 |

Smoothed first foot environment contact occurs at 0.314 s, at height 2.448 cm, tilt 3.00 degrees, and angular speed 1.09 rad/s. The previous 2 ms sample is contact-free. At 0.316 s, angular speed is 26.11 rad/s; by 0.350 s it is 223.93 rad/s. Before first contact, maximum tilt is 20.69 degrees and maximum instantaneous angular speed is 18.90 rad/s. Identity first contacts at 1.426 s.

Applied smoothed power minus identity averages [-0.014627, -0.018925]. Evaluating the original unfiltered decoder on the *recorded smoothed-run inputs* accounts for [-0.013799, -0.019264] of this difference; applying the filter to those same inputs contributes [-0.000828, 0.000339]. This is an exact algebraic decomposition, not a causal separation of feedback from filter delay. It makes explicit that most of the later change need not be the filter directly suppressing the same original signal: the entire sensorimotor trajectory has changed.

The smoothed run terminates at 0.746 s with excessive rotation, best qualified bout 0.042 s, versus identity 1.830 s and 0.412 s. Smoothing alters response delay and bilateral balance as well as high-frequency variation. This result cannot establish that CNS noise is irrelevant. Learned steering is unchanged as an algorithm but can differ between runs through changed neural input. No other control is evaluated here.

Power uses the 300 command intervals [0, 0.3 s); body summaries include the initial observation and 150 observations every 2 ms through 0.3 s. Contact event order is limited to these recorded observations. Fixed 20 ms RMS and 50 ms height windows in the JSON are descriptive, rather than a reimplementation of all scoring criteria.

[Full result](smoothing-postmortem.json); [reproducible script](smoothing-postmortem.mjs). 20,608 original decoder output values, 123,648 EMA values, and 15,456 unchanged steering values passed. No physics or neural steps were run by this analysis.
