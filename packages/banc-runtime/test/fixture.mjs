export function fixture({n=4,edges=[],gaps=[],graded=[],overrides={}}={}){
  const receptors=[[.3,3,0],[.5,8,-75],[1,6,-75],[.2,4,-75],[.5,5,0],[50,500,0],[50,400,0],[50,600,0],[50,400,0]];
  const params=new Float32Array(n*16);
  for(let i=0;i<n;i++){params.set([20,1,-60,-45,-60,2,100,.2,graded.includes(i)?1:0,-45,10,0,0,0,0,0],i*16);for(const[k,v]of Object.entries(overrides[i]||{}))params[i*16+Number(k)]=v;}
  const sorted=edges.toSorted((a,b)=>a.post-b.post),electrical=gaps.flatMap(g=>[{source:g.a,post:g.b,weight:g.weight},{source:g.b,post:g.a,weight:g.weight}]).sort((a,b)=>a.post-b.post);
  const packed=new Uint32Array((sorted.length+electrical.length)*4),f=new Float32Array(packed.buffer),offsets=new Uint32Array((n+1)*2);
  let at=0;
  for(let i=0;i<n;i++){offsets[i]=at;for(const edge of sorted.filter(e=>e.post===i)){packed[at*4]=edge.source;f[at*4+1]=edge.weight;packed[at*4+2]=edge.receptor??0;packed[at*4+3]=edge.delay??4;at++;}}
  offsets[n]=at;
  for(let i=0;i<n;i++){offsets[n+1+i]=at;for(const edge of electrical.filter(e=>e.post===i)){packed[at*4]=edge.source;f[at*4+1]=edge.weight;at++;}}
  offsets[offsets.length-1]=at;
  return {manifest:{schema:1,dataset:'BANC',materialization:888,fixture:true,neuron_count:n,dt_ms:.5,delay_slots:32,chemical_edges:sorted.length,receptors:receptors.map(([rise_ms,decay_ms,reversal_mv])=>({rise_ms,decay_ms,reversal_mv}))},offsets,edges:packed,params};
}
