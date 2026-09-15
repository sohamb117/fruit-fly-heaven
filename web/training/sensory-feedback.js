import {SensoryEncoder} from '../sensory-encoder.js';
import {validateTrainingVisionConfig} from './config-schema.js';
import {createRetinalSensor} from './retinal-sensor.js';
import {createCompactVision} from './compact-vision.js';
import {createAntennaPopulation,createAntennaAirflowModel,createNativeAntennaKinematics} from '../banc-antenna.js';
import {createLegProprioceptionPopulation,createLegProprioceptionMapper,createLegJointDescriptors,validateLegProprioceptionConfig} from '../banc-leg-proprioception.js';

const finiteVector=(x,n)=>x?.length===n&&Array.from(x).every(Number.isFinite);
export function worldVectorToRoot(vector,quaternion){
  if(!finiteVector(vector,3)||!finiteVector(quaternion,4))throw new Error('Invalid root-frame vector');
  const [w,x,y,z]=quaternion,[vx,vy,vz]=vector;
  return [(1-2*(y*y+z*z))*vx+2*(x*y+w*z)*vy+2*(x*z-w*y)*vz,
    2*(x*y-w*z)*vx+(1-2*(x*x+z*z))*vy+2*(y*z+w*x)*vz,
    2*(x*z+w*y)*vx+2*(y*z-w*x)*vy+(1-2*(x*x+y*y))*vz];
}

export function trainingSensoryScene(world,body,sceneProfile){
  return {bodyTime:body.time,position:Array.from(body.data.qpos.slice(0,3)),quaternion:Array.from(body.data.qpos.slice(3,7)),
    food:world.habitat.fruit.map(f=>({kind:f.kind,position:[f.x/10,f.z/10,f.y/10],radiusCm:f.radius/10,
      lengthCm:(f.length||0)/10,rotation:f.angle||0,path:f.path?.map(([x,y])=>[x/10,y/10,f.y/10]),remaining:f.remaining})),
    bowl:sceneProfile?{radiusCm:sceneProfile.radiusCm,floor:{baseCm:.15,radialCoefficientPerCm:.037,capRadiusCm:sceneProfile.floorCapRadiusCm},ceilingCm:sceneProfile.ceilingCm}:
      {radiusCm:6.5,floor:{baseCm:.15,radialCoefficientPerCm:.037},ceilingCm:world.habitat.ceiling/10}};
}

/** Opt-in transducers; only sensory currents enter BANC. They never consume
 * a reward, desired attitude, teacher action, or decoder output. Each episode
 * owns its visual history and virtual passive antenna state. */
export async function createTrainingSensoryResources({config,base,sensory,groups,tasteMapper,projection=null,ids=null,sceneProfile=null,legCatalog=null}){
  const visual=validateTrainingVisionConfig(config),airflow=config.antennaFeedback;
  if(visual){
    if(!projection)throw new Error('Training retinal feedback requires a projection');
    if(projection.neuron_count!==base.manifest.neuron_count||projection.ids_sha256!==base.manifest.files['ids.bin'].sha256)
      throw new Error('Training visual projection identity mismatch');
  }
  if(airflow!==undefined&&(!airflow||airflow.schema!==1||!finiteVector(airflow.windWorldCmPerSecond,3)))throw new Error('Invalid training airflow profile');
  const visualMapping=visual?{...projection,cells:projection.cells.filter(c=>/^T[45][abcd]$/.test(c.type))}:null;
  if(visual&&!visualMapping.cells.length)throw new Error('No mapped motion-sensitive BANC cells');
  const manifest=visual?{...sensory,vision:{...sensory.vision,width:visual.width,height:visual.height}}:sensory;
  const population=airflow?await createAntennaPopulation({...base,ids},sensory):null;
  const legConfig=config.legProprioception===undefined?null:validateLegProprioceptionConfig(config.legProprioception);
  const legPopulation=legConfig?await createLegProprioceptionPopulation({...base,ids},sensory,legCatalog):null;
  const makeEncoder=environment=>new SensoryEncoder(manifest,groups,environment,{tasteMapper,visualMapping});
  const indices=makeEncoder({odor:()=>0}).indices;
  return {indices,visionEnabled:!!visual,antennaEnabled:!!airflow,legEnabled:!!legConfig,
    create({world,body,fly}){
      const antennaKinematics=airflow?.mechanics?.schema===2?createNativeAntennaKinematics({mj:world.mj,model:world.model,metadata:world.metadata}):null;
      const legs=legConfig?createLegProprioceptionMapper(legPopulation,legConfig,createLegJointDescriptors(world.metadata)):null;
      const encoder=makeEncoder(world.habitat),retina=visual?createRetinalSensor({...visual.camera,width:visual.width,height:visual.height}):null,
        motion=visual?createCompactVision({...visual.motion,mapping:visualMapping,width:visual.width,height:visual.height}):null,
        antenna=airflow?createAntennaAirflowModel(population,airflow.mechanics,antennaKinematics?.geometry):null;
      if(airflow){
        const nativeWind=world.model?.opt?.wind??[0,0,0];
        if(!finiteVector(nativeWind,3)||airflow.windWorldCmPerSecond.some((x,i)=>Math.abs(x-nativeWind[i])>1e-9))
          throw new Error('Antennal airflow must match native physical wind');
      }
      const positions=new Map(Array.from(encoder.indices,(id,k)=>[id,k]));
      let lastFrame=null,frameCount=0,nextFrameMs=0,elapsedMs=0,updates=0;
      return {encoder,retina,motion,antenna,antennaKinematics,legs,
        get lastFrame(){return lastFrame;},
        update(){
          const started=performance.now(),timeMs=body.time*1000;
          if(retina&&timeMs+1e-7>=nextFrameMs){
            lastFrame=retina.render(trainingSensoryScene(world,body,sceneProfile));motion.update(lastFrame);frameCount++;
            nextFrameMs=(Math.floor((timeMs+1e-7)/visual.frameIntervalMs)+1)*visual.frameIntervalMs;
          }
          // Validate structural leg input before committing the encoder cache.
          // A corrected same-time native sample must remain retryable.
          const legValues=legs?.sample(fly.feedback,{enabled:true});
          const antennaBefore=antenna?.snapshot();let antennaValues,encoded;
          try{
            if(antenna){
              const quaternion=Array.from(body.data.qpos.slice(3,7));
              antenna.advance(antennaKinematics?antennaKinematics.sample(body.data,{bodyTimeSeconds:body.time,windWorldCmPerSecond:airflow.windWorldCmPerSecond}):
                {bodyTimeSeconds:body.time,velocityRootCmPerSecond:worldVectorToRoot(body.data.qvel.slice(0,3),quaternion),
                windRootCmPerSecond:worldVectorToRoot(airflow.windWorldCmPerSecond,quaternion)});
              antennaValues=antenna.sample({enabled:true});
            }
            encoded=encoder.update(fly,lastFrame,{odor:true,taste:true,vision:!!visual,luminance:!visual,bodySense:true,graded:motion});
          }catch(error){if(antennaBefore)antenna.restore(antennaBefore);throw error;}
          if(legs){
            const values=legValues;
            if(encoded)for(let k=0;k<values.indices.length;k++){
              const offset=positions.get(values.indices[k]);if(offset===undefined)throw new Error('Unregistered leg sensory index');
              encoded.ratesHz[offset]=values.ratesHz[k];
            }
            for(const channel of manifest.channels.filter(c=>/^self_motion_(left|right)$/.test(c.key)))
              encoder.sample.body.rates[channel.key]=channel.indices.reduce((sum,index)=>sum+encoder.ratesHz[positions.get(index)],0)/channel.indices.length;
            encoder.sample.legProprioception=values.diagnostics;
          }
          if(antenna){
            const values=antennaValues;
            if(encoded)for(let k=0;k<values.indices.length;k++){
              const offset=positions.get(values.indices[k]);if(offset===undefined)throw new Error('Unregistered antennal sensory index');
              encoded.ratesHz[offset]=values.ratesHz[k];
            }
            for(const channel of manifest.channels.filter(c=>/^antenna_(left|right)$/.test(c.key)))
              encoder.sample.body.rates[channel.key]=channel.indices.reduce((sum,index)=>sum+encoder.ratesHz[positions.get(index)],0)/channel.indices.length;
            encoder.sample.antennaFeedback=values.diagnostics??values.snapshot??antenna.snapshot?.();
          }
          if(visual){
            // The old eye means describe disabled L1/L2/L3 injection. Keep that
            // summary explicit; the active means are requested T4/T5 rates.
            // A cached encoder update must not recursively nest this summary.
            const luminanceSummary=encoder.sample.vision.luminanceSummary??{...encoder.sample.vision};
            encoder.sample.vision={...encoder.sample.vision,ready:motion.summary.ready,
              leftHz:motion.summary.leftHz??0,rightHz:motion.summary.rightHz??0,rateSource:'requested T4/T5 rates',
              luminanceSummary,resolution:[visual.width,visual.height],frameIntervalMs:visual.frameIntervalMs,
              injectionBoundary:'T4/T5 only',luminanceInputEnabled:false,optics:lastFrame?.summary,motion:motion.summary};
          }
          elapsedMs+=performance.now()-started;updates++;
          return encoded;
        },
        summary(){return {visionEnabled:!!visual,antennaEnabled:!!airflow,
          ...(legConfig?{legProprioception:{profile:legConfig.profile,selection:legPopulation.selection,polarityMode:legConfig.polarityMode}}:{}),
          ...(antennaKinematics?{antennaProfile:airflow.mechanics.profile}:{}),frameCount,updates,executionMs:elapsedMs,
          resolution:visual?[visual.width,visual.height]:null,frameIntervalMs:visual?.frameIntervalMs??null};},
        dispose(){retina?.dispose?.();motion?.dispose?.();}
      };
    }
  };
}
