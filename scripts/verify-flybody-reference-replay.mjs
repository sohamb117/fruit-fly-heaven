// Replay original native controls in the exact exported upstream flight task.
// This tests engine and observation parity, not an autonomous BANC controller.
import fs from 'node:fs/promises';
import {createHash} from 'node:crypto';
import loadMujoco from '../packages/flybody-runtime/node_modules/@mujoco/mujoco/mujoco.js';

async function replay(load, xml, metadata, fixtures) {
  const mj = await load(), model = mj.MjModel.from_xml_string(xml), data = new mj.MjData(model);
  const maxima = {qpos: 0, qvel: 0, sensorMean: 0, observation: 0, externalForce: 0};
  const frames = [];
  const maxError = (actual, expected) => expected.reduce((m, value, i) => Math.max(m, Math.abs(value - actual[i])), 0);
  const flatten = value => Array.isArray(value) ? value.flat(Infinity) : Array.from(value);
  const multiply = (a, b) => [a[0]*b[0]-a[1]*b[1]-a[2]*b[2]-a[3]*b[3], a[0]*b[1]+a[1]*b[0]+a[2]*b[3]-a[3]*b[2], a[0]*b[2]-a[1]*b[3]+a[2]*b[0]+a[3]*b[1], a[0]*b[3]+a[1]*b[2]-a[2]*b[1]+a[3]*b[0]];
  try {
    for (const key of ['qpos', 'qvel', 'ctrl', 'act', 'qacc_warmstart']) data[key].set(fixtures.initial_state[key]);
    data.time = fixtures.initial_state.time;
    mj.mj_forward(model, data);
    for (const frame of fixtures.frames) {
      const step = frame.step, ghost = metadata.ghost, ref = metadata.reference.root_qpos[step];
      data.qpos.set(ref, ghost.qpos);
      for (let i = 0; i < 3; i++) data.qpos[ghost.qpos+i] += ghost.offset[i];
      data.qvel.set(metadata.reference.qvel[step], ghost.dof);
      data.ctrl.set(frame.ctrl);
      const means = new Float64Array(9);
      for (let substep = 0; substep < 4; substep++) {
        mj.mj_step2(model, data);
        mj.mj_step1(model, data);
        // Same lazy refresh as dm_control's first bound sensor read after the
        // upstream task wrote its ghost pose and marked physics dirty.
        if (substep === 0) mj.mj_forward(model, data);
        for (let i = 0; i < 9; i++) means[i] += data.sensordata[i] / 4;
      }
      const obs = metadata.observations, matrix = data.xmat.subarray(9*obs.world_zaxis_body_id, 9*obs.world_zaxis_body_id+9);
      const quaternion = Array.from(data.qpos.subarray(3, 7)), norm = quaternion.reduce((s, x) => s+x*x, 0);
      const reciprocal = quaternion.map((x, i) => (i ? -x : x) / norm);
      const displacement = [], relativeQuaternion = [];
      for (let preview = 0; preview < 6; preview++) {
        const target = metadata.reference.root_qpos[step+1+preview];
        const delta = target.slice(0, 3).map((x, i) => x-data.qpos[i]);
        displacement.push([0, 1, 2].map(i => delta[0]*matrix[i]+delta[1]*matrix[3+i]+delta[2]*matrix[6+i]));
        relativeQuaternion.push(multiply(reciprocal, target.slice(3)));
      }
      const observation = {
        'walker/accelerometer': means.slice(0, 3),
        'walker/gyro': means.slice(3, 6),
        'walker/velocimeter': means.slice(6, 9),
        'walker/actuator_activation': data.act,
        'walker/joints_pos': obs.qpos_indices.map(i => data.qpos[i]),
        'walker/joints_vel': obs.qvel_indices.map(i => data.qvel[i]),
        'walker/world_zaxis': Array.from(matrix.slice(6, 9)),
        'walker/ref_displacement': displacement,
        'walker/ref_root_quat': relativeQuaternion,
      };
      const errors = {qpos: maxError(data.qpos, frame.state_after.qpos), qvel: maxError(data.qvel, frame.state_after.qvel)};
      for (const [key, expected] of Object.entries(frame.observation_after)) {
        const error = maxError(flatten(observation[key]), flatten(expected));
        errors[key] = error;
        maxima.observation = Math.max(maxima.observation, error);
        if (['walker/accelerometer', 'walker/gyro', 'walker/velocimeter'].includes(key)) maxima.sensorMean = Math.max(maxima.sensorMean, error);
      }
      maxima.qpos = Math.max(maxima.qpos, errors.qpos);
      maxima.qvel = Math.max(maxima.qvel, errors.qvel);
      for (const field of ['xfrc_applied', 'qfrc_applied']) for (const value of data[field]) maxima.externalForce = Math.max(maxima.externalForce, Math.abs(value));
      frames.push({step: step+1, seconds: data.time, errors});
    }
    return {passed: maxima.qpos < 1e-8 && maxima.qvel < 1e-6 && maxima.observation < 1e-5 && maxima.externalForce === 0, steps: frames.length, seconds: data.time, maxima, frames, dimensions: {nq: model.nq, nv: model.nv, nu: model.nu}};
  } finally {
    data.delete();
    model.delete();
  }
}

const [xml, metadata, fixtures] = await Promise.all([
  fs.readFile('models/flybody-flight-reference.xml', 'utf8'),
  fs.readFile('models/flybody-flight-reference.json', 'utf8').then(JSON.parse),
  fs.readFile('models/flybody-flight-reference-fixtures.json', 'utf8').then(JSON.parse),
]);
const xmlSha256 = createHash('sha256').update(xml).digest('hex');
if (xmlSha256 !== fixtures.xml_sha256 || xmlSha256 !== metadata.source.xml_sha256) throw new Error('Reference XML checksum mismatch');
const report = {
  date: new Date().toISOString(),
  scope: 'Ten native full-control steps replayed in unmodified upstream flight model dynamics, without a learned controller call or BANC. One physical fly plus the upstream non-colliding ghost reference.',
  xmlSha256,
  restore: 'Restore qpos, qvel, ctrl, act, time and qacc_warmstart; mj_forward. Write only ghost reference pose/velocity at each control interval; never correct physical root pose.',
  stepping: metadata.stepping,
  checks: {},
};
report.checks.nodeWasm = await replay(loadMujoco, xml, metadata, fixtures);
if (!process.argv.includes('--node-only')) {
  const {chromium} = await import(process.env.PLAYWRIGHT_MODULE || '/tmp/fruit-fly-browser-tests/node_modules/playwright/index.mjs');
  const browser = await chromium.launch({executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome', headless: true});
  try {
    const page = await browser.newPage();
    const base = process.env.FLY_URL || 'http://127.0.0.1:7842';
    // An inert same-origin document avoids running the production simulation.
    await page.goto(`${base}/body-model/flybody-flight-reference.xml`);
    report.checks.chromiumWasm = await page.evaluate(async ({source, xml, metadata, fixtures, base}) => {
      const {default: load} = await import(`${base}/body-engine/mujoco.js`);
      return await (0, eval)(`(${source})`)(load, xml, metadata, fixtures);
    }, {source: replay.toString(), xml, metadata, fixtures, base});
  } finally {
    await browser.close();
  }
}
report.passed = Object.values(report.checks).every(check => check.passed);
await fs.writeFile('reports/flybody-reference-replay.json', JSON.stringify(report, null, 2)+'\n');
console.log(JSON.stringify({passed: report.passed, checks: Object.fromEntries(Object.entries(report.checks).map(([key, value]) => [key, {passed: value.passed, seconds: value.seconds, maxima: value.maxima}]))}, null, 2));
if (!report.passed) process.exitCode = 1;
