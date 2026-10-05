// Development-only adapter. The production worker/controller and dynamics are
// reused unchanged; capture is enabled only by a separately pinned local plan.
import {createTrainingWorkerController} from '../training/worker.js';
import {createTrainingEnvironment} from '../training/environment.js';

const requiredAssets = ['/test/training-motor-capture-worker.js','/test/training-motor-capture.js','/test/training-safari-check.html'];
const sha = async bytes => Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', bytes)), value => value.toString(16).padStart(2, '0')).join('');
let captureSettings = null;
async function checkObserverSources() {
  if (!['127.0.0.1','localhost'].includes(self.location.hostname)) throw new Error('Motor capture worker is loopback-only');
  if (!captureSettings || Object.keys(captureSettings.assets || {}).length !== requiredAssets.length ||
    requiredAssets.some(url => !/^[a-f0-9]{64}$/.test(captureSettings.assets[url]))) throw new Error('Motor capture observer sources are not pinned');
  for (const url of requiredAssets) {
    const response = await fetch(url, {cache:'no-store'});
    if (!response.ok || await sha(await response.arrayBuffer()) !== captureSettings.assets[url]) throw new Error('Motor capture source changed: ' + url);
  }
}
const controller = createTrainingWorkerController({postMessage: message => self.postMessage(message), close: () => self.close(),
  createEnvironment: async (config, options) => {
    await checkObserverSources();
    const helperUrl = '/test/training-motor-capture.js';
    const {createMotorReplayCapture} = await import(helperUrl + '?sha256=' + captureSettings.assets[helperUrl]);
    const response = await fetch('/body-model/flybody-mujoco.xml', {cache:'no-store'});
    if (!response.ok) throw new Error('Capture model XML unavailable');
    const sourceXml = await response.text();
    const environment = await createTrainingEnvironment(config, options);
    if (await sha(new TextEncoder().encode(sourceXml)) !== environment.config.assets['/body-model/flybody-mujoco.xml']) {
      environment.dispose(); throw new Error('Capture model XML differs from pinned runtime');
    }
    return {...environment, async evaluate(job, evaluationOptions) {
      await checkObserverSources();
      const capture = createMotorReplayCapture({sourceXml, seconds:captureSettings.seconds, bodyBlockMs:environment.config.bodyBlockMs,
        sourceAssets:captureSettings.assets});
      const result = await environment.evaluate(job, {...evaluationOptions, onInitialState:capture.onInitialState, onPhysicsStep:capture.onPhysicsStep});
      const motorReplay = capture.finish(result);
      await checkObserverSources();
      return {...result, motorReplay};
    }};
  }});
self.onmessage = event => {
  if (event.data?.type === 'initialize') captureSettings = structuredClone(event.data.motorReplayCapture);
  void controller.handle(event.data);
};
