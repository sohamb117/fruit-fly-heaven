import test from 'node:test';
import assert from 'node:assert/strict';
import {upwardContactForce,measureFootSupport} from '../training/contact-observation.js';
test('contact frame, geom ordering and tangential components determine upward support',()=>{
 const upward=[0,0,1,1,0,0,0,1,0];
 assert.equal(upwardContactForce(upward,[2,0,0],true),2);
 assert.equal(upwardContactForce(upward,[2,0,0],false),0);
 assert.equal(upwardContactForce([0,0,-1,1,0,0,0,1,0],[2,0,0],false),2);
 assert.equal(upwardContactForce([1,0,0,0,0,1,0,1,0],[8,.5,0],true),.5);
});
test('body and wing collisions cannot supply foot support; contacts on one foot count once',()=>{
 const rows=[{geom:[0,1],force:4,dist:.001},{geom:[0,1],force:2,dist:.001},{geom:[0,2],force:100,dist:0},{geom:[0,3],force:200,dist:-.01},{geom:[1,2],force:50,dist:0},{geom:[0,1],force:0,dist:.001}];
 let current=0,deleted=0;const buffer={GetView:()=>[rows[current].force,0,0]};
 const body={metadata:{mass_g:1/981},geomBodyIds:[0,1,2,3],tasteBodyToLeg:[-1,0,-1,-1],model:{},contactForce:buffer,
 data:{ncon:rows.length,contact:{get(i){current=i;return{...rows[i],frame:[0,0,1,1,0,0,0,1,0],delete(){deleted++;}};},delete(){deleted++;}}},mj:{mj_contactForce(){}}};
 assert.deepEqual(measureFootSupport(body),{footSupportCount:1,footSupportFraction:6,environmentContacts:4});assert.equal(deleted,rows.length+1);
});
