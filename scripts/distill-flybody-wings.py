"""Rebuild the wing lookup with moments referenced to whole-body COM.

The original measured force-versus-power curve and oscillator frequency remain
in the model metadata. Only the force moment reference is corrected. Run with
uv run --no-project --with scipy --with mujoco==3.13.0, then run
calibrate-flybody-steering.py and prepare-flybody-runtime.py as before.
"""
from pathlib import Path
import runpy
import sys

if __name__ == '__main__':
    fitter = Path(__file__).with_name('fit-flybody-wing-interface.py')
    sys.argv = [str(fitter), '--output=models/flybody-wing-actuation.json',
                '--report=reports/flybody-wing-actuation-fit.json', *sys.argv[1:]]
    runpy.run_path(str(fitter), run_name='__main__')
