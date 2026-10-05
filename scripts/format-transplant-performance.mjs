// Format a completed benchmark without changing its measured values or gates.
// node scripts/format-transplant-performance.mjs reports/result.json [reports/result.md]
import fs from 'node:fs/promises';
import path from 'node:path';
import {median} from './benchmark-transplant-metrics.mjs';

const [input,output]=process.argv.slice(2);
if(!input)throw new Error('Usage: node scripts/format-transplant-performance.mjs INPUT.json [OUTPUT.md]');
const report=JSON.parse(await fs.readFile(input,'utf8'));
const contextPath=input.replace(/\.json$/,'-context.json');
let context;
try{context=JSON.parse(await fs.readFile(contextPath,'utf8'));}
catch(error){if(error.code!=='ENOENT')throw error;}
const format=(value,digits=5)=>Number.isFinite(value)?value.toFixed(digits):'unavailable';
const comparisons=report.comparisons||[];
const complete=comparisons.length>0&&comparisons.every(c=>c.eligible)&&report.sourceHashesUnchangedAtEnd===true;
const passed=complete&&comparisons.every(c=>c.withinOneOrderOfMagnitude===true&&c.everyPairedTrialWithinOneOrderOfMagnitude===true);
const valid=report.samples.filter(s=>s.valid).length;
const requested=report.requested||{};
const rows=comparisons.map(c=>{
  const frames=median(report.samples.filter(s=>s.valid&&s.dataset==='banc'&&s.population===c.population&&s.mode===c.mode).map(s=>s.frameHz));
  const range=c.pairedSlowdownRange?.map(v=>format(v,2)).join('–')||'unavailable';
  const simSeconds=Number.isFinite(c.bancMedianSpeed)?c.bancMedianSpeed*requested.seconds:null;
  return `| ${c.population} | ${c.mode} | ${format(c.flywireMedianSpeed)}× | ${format(c.bancMedianSpeed)}× | ${format(c.slowdown,2)}× | ${range}× | ${format(simSeconds,3)} s | ${format(frames,1)} |`;
});
const invalid=report.samples.filter(s=>!s.valid).map(s=>`- ${s.dataset}, ${s.mode}, population ${s.population}, trial ${s.trial}: ${[s.error,...(s.validityIssues||[])].filter(Boolean).join('; ')||'invalid sample'}`);
const source=path.basename(input);
const result=passed
  ?'The current BANC build meets the factor-of-ten throughput threshold against the retained FlyWire system in every accepted mode and paired trial.'
  :complete
    ?'The current BANC build does not meet the factor-of-ten throughput threshold in every accepted mode and paired trial.'
    :'This run is not eligible for the requested throughput gate because measurements or source-consistency checks are incomplete.';
const markdown=[
  '# Matched whole-product throughput',
  '',
  `${result} ${valid}/${report.samples.length} measurements passed the harness checks. Source hashes ${report.sourceHashesUnchangedAtEnd===true?'were unchanged through the measurement run':'were not verified unchanged'}. This measures execution speed; it does not establish successful behavior or real-time simulation.`,
  '',
  ...(context?.build?[`${context.build}. [Measurement context](${path.basename(contextPath)}).`,'']:[]),
  `${requested.trials} ${requested.seconds}-second measurement windows per mode and dataset, using the original 3D UI with binocular rendering, FlyVis, color, every sensory switch, direct motor coupling, the neural body clock and open anatomy inspection. FlyWire retains its kinematic body; BANC uses muscles and synchronized native MuJoCo. These are the intended product workloads, not equal numerical workloads.`,
  '',
  `| Flies | Mode | FlyWire neural/real time | BANC neural/real time | BANC slowdown | Paired slowdown range | BANC sim time per ${requested.seconds} s wall | BANC callbacks/s |`,
  '|---:|---|---:|---:|---:|---:|---:|---:|',
  ...rows,
  '',
  'Throughput is simulated neural milliseconds divided by elapsed wall-clock milliseconds. The table also converts each BANC rate into simulated seconds per measurement window. These rates remain below real time even when the UI redraws frequently. The callback column counts requestAnimationFrame callbacks and is not a guarantee of distinct presented frames. Renderer and anatomy liveness are checked separately.',
  '',
  `Measured on ${report.cpu} (${report.platform}/${report.arch}), starting ${report.date}. Short paired windows are subject to shared-machine load and trial variation. A prior report is not a controlled before/after performance experiment.`,
  '',
  ...(invalid.length?['Invalid samples:', '',...invalid,'']:[]),
  ...(report.changedSourceFiles?.length?['Changed source files:', '',...report.changedSourceFiles.map(f=>`- ${f}`),'']:[]),
  `[Complete measurements, effective model settings, source hashes and validity checks](${source}).`,
  '',
  'Reproduce:',
  '',
  '```sh',
  `PLAYWRIGHT_MODULE=/absolute/path/to/playwright/index.mjs node scripts/benchmark-transplant.mjs --populations=${requested.populations?.join(',')} --modes=${requested.modes?.join(',')} --datasets=${requested.datasets?.join(',')} --seconds=${requested.seconds} --trials=${requested.trials} --output=${input}`,
  '```',
  '',
].join('\n');
if(output)await fs.writeFile(output,markdown);
else process.stdout.write(markdown);
