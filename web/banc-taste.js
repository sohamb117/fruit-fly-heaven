// Anatomical routing only: each annotated sugar GRN receives the contact of
// its own organ/side. 150 Hz remains the existing, uncalibrated stimulus prior.
export const BANC_TASTE_PRIORS=Object.freeze({contactRateHz:150,
  rateStatus:'Existing stimulus-rate prior, not measured receptor physiology.',
  anatomyStatus:'BANC v888 body_part_sensory and side annotations; missing identity or laterality abstains.',
  legOrder:Object.freeze(['front_left','middle_left','hind_left','front_right','middle_right','hind_right']),
  bilateralOrder:Object.freeze(['left','right'])});

export function createBancTasteMapper(sensory,sweet){
  if(!Array.isArray(sensory)||!sweet||!Array.from(sweet).every(i=>Number.isInteger(i)&&i>=0))
    throw new Error('Invalid BANC taste annotations or sugar indices');
  const annotations=new Map();
  for(const row of sensory){
    if(annotations.has(row.index))throw new Error(`Duplicate sensory annotation for ${row.index}`);
    annotations.set(row.index,row);
  }
  const mapped=new Map(),unmapped=[],organs={labellum:0,wing_margin:0,front_leg:0,middle_leg:0,hind_leg:0};
  const indices=Array.from(new Set(sweet));
  for(const index of indices){
    const row=annotations.get(index),side=row?.side==='left'?0:row?.side==='right'?1:null;
    let reason=null,field=null,slot=null;
    if(!row)reason='missing sensory annotation';
    else if(row.kind!=='taste')reason='annotation is not taste';
    else if(side===null)reason='missing or unsupported laterality';
    else if(row.body_part==='labellum'){field='mouthFoodContact';slot=side;}
    else if(row.body_part==='wing_margin'){field='wingFoodContact';slot=side;}
    else if(['front_leg','middle_leg','hind_leg'].includes(row.body_part)){
      field='legFoodContact';slot=['front_leg','middle_leg','hind_leg'].indexOf(row.body_part)+3*side;
    }else reason='missing or unsupported sensory organ';
    if(reason)unmapped.push({index,reason,side:row?.side??null,bodyPart:row?.body_part??null,cellType:row?.cell_type??null});
    else{mapped.set(index,{field,slot});organs[row.body_part]++;}
  }
  return {coverage:{requested:indices.length,mapped:mapped.size,unmapped,organs,
    evidence:'Only supplied BANC organ and side annotations are used. Unknown laterality is never assigned from another cell or generic body contact.'},
    rate(index,feedback,enabled=true){
      if(enabled!==true)return 0;
      const target=mapped.get(index);
      if(!target)return 0;
      const contacts=feedback?.[target.field],length=target.field==='legFoodContact'?6:2;
      const array=Array.isArray(contacts)||ArrayBuffer.isView(contacts),contact=contacts?.[target.slot];
      return array&&contacts.length===length&&(contact===true||contact===1)?BANC_TASTE_PRIORS.contactRateHz:0;
    }};
}
