import {test} from 'node:test';
import assert from 'node:assert/strict';
import {createBancTasteMapper} from '../banc-taste.js';

const organs=['labellum','wing_margin','front_leg','middle_leg','hind_leg'];
const sensory=organs.flatMap((body_part,k)=>['left','right'].map((side,j)=>({index:k*2+j,kind:'taste',body_part,side,cell_type:'annotated_sugar'})));
const sweet=sensory.map(s=>s.index),mapper=createBancTasteMapper(sensory,sweet);
const empty=()=>({mouthFoodContact:[false,false],wingFoodContact:[false,false],legFoodContact:Array(6).fill(false)});
const active=(m,f)=>sweet.filter(i=>m.rate(i,f)>0);

test('each actual organ and side contact drives exactly its annotated sugar cells',()=>{
  for(let i=0;i<10;i++){
    const f=empty(),field=i<2?'mouthFoodContact':i<4?'wingFoodContact':'legFoodContact',slot=i<4?i%2:Math.floor((i-4)/2)+3*(i%2);
    f[field][slot]=true;assert.deepEqual(active(mapper,f),[i]);assert.equal(mapper.rate(i,f),150);
  }
});

test('standing on fruit cannot stimulate the mouth or wing sugar cells without their own contact',()=>{
  const f=empty();f.legFoodContact[0]=true;f.contact=true;f.onFood=true;f.mouthContact=true;
  assert.deepEqual(active(mapper,f),[4]);
});

test('taste switch removes every organ input and feedback is read only',()=>{
  const f=Object.freeze({mouthFoodContact:Object.freeze([true,true]),wingFoodContact:Object.freeze([true,true]),legFoodContact:Object.freeze(Array(6).fill(true))});
  for(const index of sweet){assert.equal(mapper.rate(index,f),150);assert.equal(mapper.rate(index,f,false),0);}
});

test('unknown laterality, missing cells and unknown organs are explicit abstentions',()=>{
  const annotations=[{index:10,kind:'taste',side:null,body_part:'labellum'},
    {index:11,kind:'taste',side:'left',body_part:'unmapped_organ'},
    {index:12,kind:'proprioception',side:'left',body_part:'front_leg'}];
  const m=createBancTasteMapper(annotations,[10,11,12,13]);
  assert.equal(m.coverage.mapped,0);assert.equal(m.coverage.unmapped.length,4);
  for(let index=10;index<=13;index++)assert.equal(m.rate(index,{mouthFoodContact:[true,true],legFoodContact:Array(6).fill(true)}),0);
  assert.match(m.coverage.unmapped.find(r=>r.index===13).reason,/missing sensory annotation/);
});

test('only requested cell identities are driven, regardless of array order or extra annotations',()=>{
  const m=createBancTasteMapper([...sensory].reverse(),[7,0,7]);const f=empty();f.mouthFoodContact[0]=true;f.legFoodContact[4]=true;
  assert.equal(m.coverage.requested,2);assert.deepEqual(active(m,f),[0,7]);assert.equal(m.rate(1,f),0);
});

test('missing, aggregate or malformed contact feedback never becomes per-side contact',()=>{
  for(const f of [undefined,{mouthFoodContact:true},{mouthFoodContact:[true]},
    {mouthFoodContact:['true',0]},{mouthFoodContact:[-1,0]},{mouthFoodContact:[1.1,0]},
    {mouthFoodContact:[true,false,false]},{contact:true}])assert.equal(mapper.rate(0,f),0);
});

test('native Uint8 contact flags and serialized exact 0/1 arrays preserve each side',()=>{
  for(const mouthFoodContact of [new Uint8Array([1,0]),[1,0]]){
    assert.equal(mapper.rate(0,{mouthFoodContact}),150);assert.equal(mapper.rate(1,{mouthFoodContact}),0);
  }
});

test('duplicate annotations and invalid indices fail rather than silently reassign a cell',()=>{
  assert.throws(()=>createBancTasteMapper([sensory[0],sensory[0]],[0]),/Duplicate/);
  assert.throws(()=>createBancTasteMapper(sensory,[-1]),/Invalid/);
});
