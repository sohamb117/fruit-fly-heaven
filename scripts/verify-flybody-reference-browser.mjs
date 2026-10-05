// Actual Chromium closed-loop run of the released learned flight reference.
import fs from 'node:fs/promises';
import {createHash} from 'node:crypto';
const {chromium} = await import(process.env.PLAYWRIGHT_MODULE || '/tmp/fruit-fly-browser-tests/node_modules/playwright/index.mjs');
const browser = await chromium.launch({executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome', headless: true});
let report;
try {
  const page = await browser.newPage();
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  const base = process.env.FLY_URL || 'http://127.0.0.1:7842';
  await page.goto(`${base}/banc/verify.html`);
  const result = await page.evaluate(async () => {
    const loadStart = performance.now();
    const [{default: loadMujoco}, {FlyBodyFlightReference}, xml, metadata, policy] = await Promise.all([
      import('/body-engine/mujoco.js'),
      import('/flybody-flight-reference.js'),
      fetch('/body-model/flybody-flight-reference.xml').then(r => r.text()),
      fetch('/body-model/flybody-flight-reference.json').then(r => r.json()),
      fetch('/body-model/flybody-flight-policy.json').then(r => r.json()),
    ]);
    const mj = await loadMujoco(), body = new FlyBodyFlightReference(mj, xml, metadata, policy);
    const loadSeconds = (performance.now()-loadStart)/1000;
    let maximumReferenceErrorCm = 0, maximumAttitudeErrorDegrees = 0, minimumHeightCm = Infinity, maximumHeightCm = -Infinity, maximumAppliedForce = 0, maximumRootActuatorForce = 0, allFinite = true;
    const poseSamples = [], steps = 5994, start = performance.now();
    try {
      for (let i = 0; i < steps; i++) {
        if (!body.step()) throw new Error(`Reference ended at step ${i}`);
        const sample = body.sample();
        allFinite &&= sample.finite;
        maximumReferenceErrorCm = Math.max(maximumReferenceErrorCm, sample.referenceError);
        maximumAttitudeErrorDegrees = Math.max(maximumAttitudeErrorDegrees, sample.attitudeError);
        minimumHeightCm = Math.min(minimumHeightCm, sample.position[2]);
        maximumHeightCm = Math.max(maximumHeightCm, sample.position[2]);
        maximumAppliedForce = Math.max(maximumAppliedForce, sample.maximumAppliedForce);
        maximumRootActuatorForce = Math.max(maximumRootActuatorForce, sample.maximumRootActuatorForce);
        if (i % 120 === 0 || i === steps-1) poseSamples.push(sample);
      }
      const wallSeconds = (performance.now()-start)/1000, seconds = body.data.time;
      const terminationReason = body.terminationReason;
      const stoppedAtLimit = metadata.episode.maxControlSteps === steps && body.done && body.step() === false && body.data.time === seconds;
      return {passed: allFinite && maximumReferenceErrorCm < .08 && maximumAttitudeErrorDegrees < 15 && minimumHeightCm > .9 && maximumHeightCm < 1.1 && maximumAppliedForce === 0 && maximumRootActuatorForce === 0 && terminationReason === 'trajectory complete' && stoppedAtLimit,
        steps, seconds, wallSeconds, simulationSecondsPerWallSecond: seconds/wallSeconds, loadSeconds, maximumReferenceErrorCm, maximumAttitudeErrorDegrees, minimumHeightCm, maximumHeightCm, maximumAppliedForce, maximumRootActuatorForce, allFinite, terminationReason, stoppedAtLimit, poseSamples, xmlSha256: metadata.source.xml_sha256};
    } finally {
      body.dispose();
    }
  });
  report = {date: new Date().toISOString(), scope: 'Actual Chromium closed-loop execution of released FlyBody learned flight controller, starting airborne on prescribed straight reference; not BANC, takeoff or landing.', browserVersion: browser.version(), userAgent: await page.evaluate(() => navigator.userAgent), errors, ...result};
  report.passed &&= errors.length === 0;
  report.sourceSha256 = Object.fromEntries(await Promise.all(['web/flybody-flight-reference.js', 'web/flybody-policy.js', 'models/flybody-flight-reference.xml', 'models/flybody-flight-reference.json', 'models/flybody-flight-policy.json'].map(async file => [file, createHash('sha256').update(await fs.readFile(file)).digest('hex')])));
} finally {
  await browser.close();
}
await fs.writeFile('reports/flybody-reference-browser.json', JSON.stringify(report, null, 2)+'\n');
console.log(JSON.stringify(Object.fromEntries(Object.entries(report).filter(([key]) => key !== 'poseSamples')), null, 2));
if (!report.passed) process.exitCode = 1;
