"""Build a portable RGB-to-opsin LUT from PBRT and drEye's published methods.

RGB reconstruction is not a spectral measurement. This model assumes no light
below 400 nm. Govardovskii templates use published pigment peak wavelengths;
screening pigments, polarization and phototransduction are not modeled.
"""
from pathlib import Path
import hashlib
import json
import re
import urllib.request
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT/'references/color-research'
OUT = ROOT/'packages/fly-color-wasm/model'
PBRT = 'https://raw.githubusercontent.com/mmp/pbrt-v3/13d871faae88233b327d04cda24022b8bb0093ee/'
DREYE = 'https://raw.githubusercontent.com/neuralsignal/dreye/b36d2870ec661a34e2cda6d9c5b23ea89eb0dabb/'
SOURCES = {'pbrt-spectrum.cpp': PBRT+'src/core/spectrum.cpp', 'pbrt-LICENSE.txt': PBRT+'LICENSE.txt',
           'dreye-filter_templates.py': DREYE+'dreye/api/filter_templates.py',
           'dreye-capture.py': DREYE+'dreye/api/capture.py', 'dreye-LICENSE': DREYE+'LICENSE'}

# Govardovskii et al. 2000 A1 alpha + beta bands, matching drEye defaults.
def sensitivities(wavelengths):
    peak = np.array([345., 375., 437., 508.])[:, None]
    w = wavelengths[None, :]
    x = peak/w
    a = .8795 + .0459*np.exp(-(peak-300)**2/11940)
    alpha = 1/(np.exp(69.7*(a-x))+np.exp(28*(.922-x))+np.exp(-14.9*(1.104-x))+.674)
    beta = .26*np.exp(-((w-(189+.315*peak))/(-40.5+.195*peak))**2)
    return alpha+beta


def main():
    REF.mkdir(parents=True, exist_ok=True); OUT.mkdir(parents=True, exist_ok=True)
    for name, url in SOURCES.items():
        if not (REF/name).exists(): (REF/name).write_bytes(urllib.request.urlopen(url).read())
    source = (REF/'pbrt-spectrum.cpp').read_text()
    def array(name):
        body = re.search(r'const Float '+name+r'\[nRGB2SpectSamples\] = \{(.*?)\};', source, re.S).group(1)
        return np.array([float(v.strip().removesuffix('f')) for v in body.split(',') if v.strip()])
    knots = array('RGB2SpectLambda')
    w = np.arange(300., 721.)
    # Use the illuminant reconstruction for rendered pixel radiance, not the
    # reflectance basis. Interpolate published samples, then explicitly zero UV.
    basis = np.array([np.interp(w, knots, array('RGBIllum2Spect'+c), left=0, right=0)
                      for c in ['White','Cyan','Magenta','Yellow','Red','Green','Blue']])
    basis[:, w < 400] = 0
    sensitivity = sensitivities(w)
    integration = np.ones(w.size); integration[[0,-1]] = .5
    # Convert relative radiometric spectra to photon counts (lambda/hc); hc
    # cancels against a unit-energy 300–720 nm reference, independently per opsin.
    filters = sensitivity*w*integration
    filters /= filters.sum(axis=1, keepdims=True)
    def catches(rgb):
        r,g,b = np.asarray(rgb).T
        coefficients = np.zeros((r.size,7))
        low = np.minimum(np.minimum(r,g),b); coefficients[:,0] = low
        # PBRT/Smits six sectors. The clamp is after spectral reconstruction.
        for mask,secondary,primary,a,c in [
            ((r<=g)&(g<=b),1,6,g-r,b-g), ((r<=b)&(b<g),1,5,b-r,g-b),
            ((g<r)&(r<=b),2,6,r-g,b-r), ((g<=b)&(b<r),2,4,b-g,r-b),
            ((b<r)&(r<=g),3,5,r-b,g-r), ((b<g)&(g<r),3,4,g-b,r-g)]:
            coefficients[mask,secondary] = a[mask]; coefficients[mask,primary] = c[mask]
        spectra = np.maximum(0, .86445*(coefficients@basis))
        return spectra@filters.T
    size = 33
    axis = np.linspace(0,1,size)
    rgb = np.stack(np.meshgrid(axis,axis,axis,indexing='ij'),axis=-1).reshape(-1,3)
    lut = np.concatenate([catches(rgb[i:i+1024]) for i in range(0,len(rgb),1024)]).astype('<f4')
    payload = lut.tobytes(); (OUT/'fly-color-lut.bin').write_bytes(payload)
    model = dict(schema_version=1, size=size, channels=['Rh3','Rh4','Rh5','Rh6'],
        layout='R-major, then G, then B, interleaved channels; Float32 little-endian',
        input='sRGB bytes decoded to linear RGB before trilinear lookup',
        method='PBRT v3 Smits-style illuminant reconstruction, nonnegative spectral clamp, Govardovskii 2000 A1 templates, photon-catch integration',
        peaks_nm=[345,375,437,508], wavelength_nm=[300,720], illumination_cutoff_nm=400,
        uv_assumption='Zero incident light below 400 nm. RGB does not recover ultraviolet.',
        normalization='Each catch divided by that opsin capture under unit-energy equal-energy light over 300–720 nm; no amplification of weak UV channels to RGB white.',
        units='relative photon capture; not volts, spikes or perceived RGB',
        limitations=['Reconstructed spectra are one RGB-compatible approximation, not measured material spectra.',
                     'Template pigment sensitivities omit ocular screening, sensitizing pigments, polarization and adaptation.',
                     'The LUT approximates the spectral integral; the brain adapter supplies separate assumed rate gains.'],
        lut_file='fly-color-lut.bin',lut_sha256=hashlib.sha256(payload).hexdigest(),
        sources={name:dict(url=url,sha256=hashlib.sha256((REF/name).read_bytes()).hexdigest()) for name,url in SOURCES.items()},
        peak_source='https://www.nature.com/articles/s41598-020-74742-1',
        template_source='https://doi.org/10.1017/S0952523800174036')
    (OUT/'fly-color-model.json').write_text(json.dumps(model,indent=2)+'\n')
    # Independent full spectral reference vectors, off the LUT lattice.
    rng = np.random.default_rng(783)
    srgb = np.vstack((np.array([[0,0,0],[255,255,255],[255,0,0],[0,255,0],[0,0,255],[128,128,128],[239,181,43]]),rng.integers(0,256,(200,3))))
    s = srgb/255; linear = np.where(s<=.04045,s/12.92,((s+.055)/1.055)**2.4)
    (OUT/'reference.json').write_text(json.dumps(dict(rgb=srgb.tolist(),captures=catches(linear).tolist()),separators=(',',':'))+'\n')
    notices = ROOT/'packages/fly-color-wasm/third-party-licenses'; notices.mkdir(parents=True,exist_ok=True)
    for name in ['pbrt-LICENSE.txt','dreye-LICENSE']:(notices/name).write_bytes((REF/name).read_bytes())
    print(json.dumps(dict(lut_bytes=len(payload),size=size,white=catches([[1,1,1]])[0].tolist(),sha256=model['lut_sha256'])))

if __name__ == '__main__': main()
