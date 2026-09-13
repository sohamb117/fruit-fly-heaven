import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {bancBodyRate,BANC_ROTATION_MODEL,hasBancContactChemosensoryFunction,hasBancUnassignedBristlePosition} from '../banc-ground-sense.js';
import {SensoryEncoder,bodyInputRates} from '../sensory-encoder.js';
const feedback=()=>({legs:Array.from({length:6},()=>({loadBodyWeights:0,collision:0,tibiaAngle:0,tibiaVelocity:0,coxaAngle:0,vibration:0})),angularVelocity:[0,0,0],wingPower:0,wingPowerLeft:0,wingPowerRight:0,halterePower:[0,0],haltereSteering:{left:{},right:{}}});
const rotation=(organ,side)=>({kind:'rotation',organ,side});
test('loading one native leg stimulates only that leg load afferents',()=>{
 const f=feedback();f.legs[4].loadBodyWeights=.3;
 const rates=f.legs.map((_,leg)=>bancBodyRate({kind:'load',leg},f));assert.deepEqual(rates,[0,0,0,0,24,0]);
 f.legs[0].tibiaVelocity=100;f.speed=100;f.yaw=100;
 assert.equal(bancBodyRate({kind:'load',leg:0},f),0);
});
test('assisted comparison retains the existing sensory adapter for its kinematic body',()=>{
 const manifest={schema_version:1,neuron_count:5,vision:{width:32,height:16,receptors:[{index:3,side:'left',u:0,v:0}]},channels:[{key:'self_motion_left',indices:[4]}],body_transducers:[{index:4,kind:'load',leg:0}]};
 const environment={surface:()=>({y:0,contact:false}),odor:()=>0},encoder=new SensoryEncoder(manifest,{odor_left:[0],odor_right:[1],sweet:[2]},environment);
 const f={legs:Array.from({length:6},()=>({support:1,angle:.2,speed:2})),speed:1,yaw:0};
 const result=encoder.update({x:0,y:0,z:0,heading:0,bodyTime:0,contact:false,feedback:f},null,{vision:false});
 assert(Math.abs(result.ratesHz[4]-bodyInputRates(f).self_motion_left)<1e-5);
});
test('airborne joint motion is sensed without inventing support or floor touch',()=>{
 const f=feedback();f.legs[0].tibiaVelocity=12;
 assert(bancBodyRate({kind:'velocity',leg:0},f)>0);
 assert.equal(bancBodyRate({kind:'touch',leg:0},f),0);assert.equal(bancBodyRate({kind:'load',leg:0},f),0);
});
test('explicit chemical function annotations do not become dry mechanical contact drive',()=>{
 const f=feedback();f.legs[0].collision=1;
 for(const functionLabel of ['sugar, low_salt, Gr64f, Ir56b','contact_pheromone, ppk23, ppk25']){
  const sensor={kind:'touch',leg:0,function:functionLabel};
  assert.equal(hasBancContactChemosensoryFunction(sensor),true);
  assert.equal(bancBodyRate(sensor,f),0);
 }
 assert.equal(hasBancContactChemosensoryFunction({function:'other,  SUGAR , Gr64f'}),true);
 assert.equal(hasBancContactChemosensoryFunction({function:'sugar_like, non_contact_pheromone'}),false,'classification requires exact function tokens');
 assert.equal(bancBodyRate({kind:'touch',leg:0,function:''},f),60,'ordinary mechanosensory bristles retain collision input');
});
test('unassigned joint-angle bristle abstains without suppressing mapped position receptors',()=>{
 const f=feedback(),unassigned={kind:'touch',leg:0,function:'joint_angle'},position={kind:'position',leg:0,function:'joint_angle'};
 f.legs[0].collision=1;
 assert.equal(hasBancUnassignedBristlePosition(unassigned),true);assert.equal(hasBancUnassignedBristlePosition(position),false);
 assert.equal(bancBodyRate(unassigned,f),0);assert.equal(bancBodyRate(position,f),5);
 f.legs[0].tibiaAngle=.3;f.legs[0].coxaAngle=.2;
 assert.equal(bancBodyRate(unassigned,f),0,'no invented joint response');assert.equal(bancBodyRate(position,f),10.5);
 assert.equal(hasBancUnassignedBristlePosition({kind:'touch',function:'other, JOINT_ANGLE '}),true);
 assert.equal(hasBancUnassignedBristlePosition({kind:'touch',function:'joint_angle_like'}),false);
});
test('prepared SNta35 identity is excluded from collision while all mapped position receptors remain',async()=>{
 const source=JSON.parse(await readFile(new URL('../../data/prepared/banc888/console/sensory-inputs.json',import.meta.url)));
 const active=source.body_transducers.filter(hasBancUnassignedBristlePosition),excluded=(source.body_transducer_exclusions||[]).filter(s=>s.function==='joint_angle');
 assert.equal(active.length+excluded.length,1);assert.equal([...active,...excluded][0].index,42986);
 for(const sensor of excluded){
  assert.equal(sensor.root_id,'720575941480808867');assert.equal(sensor.cell_type,'SNta35');
  assert.match(sensor.reason,/Joint identity and tuning are unassigned/);
  assert(!source.channels.some(channel=>channel.indices.includes(sensor.index)));
 }
 const f=feedback();f.legs[0].collision=1;
 for(const sensor of active)assert.equal(bancBodyRate(sensor,f),0);
 const positions=source.body_transducers.filter(s=>s.kind==='position');assert.equal(positions.length,403);
 for(const sensor of positions)assert.equal(bancBodyRate(sensor,f),5);
});
test('prepared chemical bristle annotations cannot bypass taste-off through native body sense',async()=>{
 const source=JSON.parse(await readFile(new URL('../../data/prepared/banc888/console/sensory-inputs.json',import.meta.url)));
 // A regenerated manifest removes these cells from the channel and records
 // them as exclusions; older artifacts are protected by the runtime guard.
 const sensors=source.body_transducers.filter(hasBancContactChemosensoryFunction);
 const exclusions=source.body_transducer_exclusions||[];
 assert.equal(sensors.length+exclusions.filter(hasBancContactChemosensoryFunction).length,10);
 const controls=source.body_transducers.filter(s=>s.kind==='touch'&&!hasBancContactChemosensoryFunction(s)&&!hasBancUnassignedBristlePosition(s)).slice(0,2);
 const testSensors=[...sensors,...controls];
 const manifest={schema_version:1,neuron_count:source.neuron_count,vision:{width:32,height:16,receptors:[{index:0,side:'left',u:0,v:0}]},
  channels:[{key:'touch_left',indices:testSensors.map(s=>s.index)}],body_transducers:testSensors};
 const encoder=new SensoryEncoder(manifest,{odor_left:[],odor_right:[],sweet:[]},{surface:()=>({y:0,contact:false}),odor:()=>0});
 const f=feedback();f.legs.forEach(leg=>leg.collision=1);
 const result=encoder.update({x:0,y:0,z:0,heading:0,bodyTime:0,contact:false,feedback:f},null,{vision:false,taste:false});
 const rates=new Map(Array.from(result.indices,(index,i)=>[index,result.ratesHz[i]]));
 for(const sensor of sensors)assert.equal(rates.get(sensor.index),0);
 for(const sensor of controls)assert.equal(rates.get(sensor.index),60);
});
test('haltere feedback responds to rotation during flight with unloaded legs',()=>{
 const f=feedback();f.halterePower[0]=.7;f.angularVelocity=[4,3,0];
 assert.equal(bancBodyRate(rotation('haltere','left'),f),17.5);
 assert.equal(bancBodyRate(rotation('haltere','right'),f),0);
 assert.equal(bancBodyRate(rotation('haltere','left'),f,false),0);
});
test('wing deployment and haltere muscle force gate only the annotated organ and side',()=>{
 const f=feedback();f.wingPower=1;f.wingPowerLeft=.8;f.angularVelocity=[3,4,0];
 assert.equal(bancBodyRate(rotation('wing_base','left'),f),20);
 assert.equal(bancBodyRate(rotation('wing_base','right'),f),0);
 assert.equal(bancBodyRate(rotation('haltere','left'),f),0);
 f.halterePower[1]=.4;
 assert.equal(bancBodyRate(rotation('haltere','right'),f),10);
 assert.equal(bancBodyRate(rotation('haltere','left'),f),0);
 assert.equal(bancBodyRate(rotation('wing_base','right'),f),0);
 delete f.wingPowerLeft;assert.equal(bancBodyRate(rotation('wing_base','left'),f),0,'mean wing power is not a side-specific fallback');
 for(const sensor of [rotation('haltere',null),rotation('wing_base','unknown'),rotation('leg','left'),{kind:'rotation'}])assert.equal(bancBodyRate(sensor,f),0);
});
test('same-side haltere steering adds basal strain only, without inventing rotation or muscle direction',()=>{
 const f=feedback(),left=rotation('haltere','left');f.haltereSteering.left.hi1_muscle=.6;
 assert.equal(bancBodyRate(left,f),3);f.angularVelocity=[100,0,0];
 assert.equal(bancBodyRate(left,f),3,'steering alone does not supply the rotation-sensitive power term');
 assert.equal(bancBodyRate(rotation('haltere','right'),f),0);
 assert.equal(bancBodyRate(rotation('wing_base','left'),f),0);
 f.haltereSteering.left.hi2_muscle=.4;assert.equal(bancBodyRate(left,f),3,'bounded max pooling does not sum unrelated steering muscle forces');
 f.haltereSteering.left={unknown_muscle:1,wing_b1_muscle:1};assert.equal(bancBodyRate(left,f),0);
});
test('rotation encoding remains explicitly unsigned, bounded and independent of a desired pose',()=>{
 const f=feedback(),left=rotation('haltere','left');f.halterePower=[1,0];
 const rates=[[4,3,0],[-4,-3,0],[0,0,5]].map(omega=>{f.angularVelocity=omega;return bancBodyRate(left,f);});
 assert.deepEqual(rates,[25,25,25]);assert.ok(BANC_ROTATION_MODEL.status.includes('unsigned'));
 f.foodBearing=-1;f.desiredHeight=100;assert.equal(bancBodyRate(left,f),25);
 f.angularVelocity=[1000,0,0];f.halterePower[0]=10;assert.equal(bancBodyRate(left,f),100);
 f.halterePower[0]=-1;assert.equal(bancBodyRate(left,f),0);
 f.halterePower[0]=NaN;f.haltereSteering.left.hi1_muscle=Infinity;assert.equal(bancBodyRate(left,f),0);
});
test('actual BANC rotation afferents preserve organ and side through the production encoder',async()=>{
 const source=JSON.parse(await readFile(new URL('../../data/prepared/banc888/console/sensory-inputs.json',import.meta.url))),sensors=source.body_transducers.filter(s=>s.kind==='rotation');
 const counts={};for(const sensor of sensors){const key=sensor.organ+':'+sensor.side;counts[key]=(counts[key]||0)+1;}
 assert.deepEqual(counts,{'haltere:left':171,'wing_base:left':62,'wing_base:right':59,'haltere:right':157});
 const manifest={schema_version:1,neuron_count:source.neuron_count,vision:{width:32,height:16,receptors:[{index:0,side:'left',u:0,v:0}]},channels:[{key:'self_motion_left',indices:sensors.map(s=>s.index)}],body_transducers:sensors};
 const encoder=new SensoryEncoder(manifest,{odor_left:[],odor_right:[],sweet:[]}, {surface:()=>({y:0,contact:false}),odor:()=>0}),f=feedback();
 f.wingPower=1;f.wingPowerRight=.8;f.halterePower[0]=.6;f.angularVelocity=[3,4,0];
 const result=encoder.update({x:0,y:0,z:0,heading:0,bodyTime:0,contact:false,feedback:f},null,{vision:false});
 const rates=new Map(Array.from(result.indices,(index,i)=>[index,result.ratesHz[i]]));
 for(const sensor of sensors){const expected=sensor.organ==='haltere'?(sensor.side==='left'?15:0):(sensor.side==='right'?20:0);assert.equal(rates.get(sensor.index),expected);}
});
test('encoder preserves per-cell ground signals inside original UI summary channels',()=>{
 const sensors=[{index:4,kind:'load',leg:0},{index:5,kind:'load',leg:1}];
 const manifest={schema_version:1,neuron_count:6,vision:{width:32,height:16,receptors:[{index:3,side:'left',u:0,v:0}]},channels:[{key:'self_motion_left',indices:[4,5]}],body_transducers:sensors};
 const environment={surface:()=>({y:0,contact:false}),odor:()=>0},encoder=new SensoryEncoder(manifest,{odor_left:[0],odor_right:[1],sweet:[2]},environment);
 const f=feedback();f.legs[0].loadBodyWeights=.25;const pose={x:0,y:0,z:0,heading:0,bodyTime:0,contact:false,feedback:f};
 const on=encoder.update(pose,null,{vision:false});assert.equal(on.ratesHz[4],20);assert.equal(on.ratesHz[5],0);
 const off=encoder.update(pose,null,{vision:false,bodySense:false});assert.equal(off.ratesHz[4],0);assert.equal(off.ratesHz[5],0);
});
