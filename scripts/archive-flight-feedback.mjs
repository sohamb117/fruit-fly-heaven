// Preserve executable sources before updating a local experimental profile.
import fs from 'node:fs/promises';
import path from 'node:path';
import {installFeedbackRuntime,root,sha} from './flight-feedback-runtime.mjs';
const input=path.resolve(process.argv[2]??path.join(root,'reports/sensory-feedback-20260915'));
const output=path.resolve(process.argv[3]??path.join(input,'v1-source-archive'));
if(!output.startsWith(path.join(root,'reports')+path.sep))throw new Error('Archive must remain inside reports');
await fs.mkdir(output,{recursive:true});
for(const variant of ['off','vision','airflow','vision-airflow']){
  const file=path.join(input,variant+'.bundle.json'),runtime=await installFeedbackRuntime(file);
  try{
    const archive=structuredClone(runtime.bundle);
    for(const url of Object.keys(runtime.config.assets))if(url.endsWith('.js'))archive.assets[url]=(await runtime.read(url)).toString('utf8');
    await runtime.verify();
    archive.sourceArchive={sourceBundleSha256:runtime.identity.bundleSha256,createdAt:new Date().toISOString(),
      javascriptSources:Object.keys(archive.assets).filter(url=>url.endsWith('.js')).length,
      externalAssets:'Graph, prepared data and native WASM binaries remain mandatory hash-verified local dependencies.'};
    const bytes=JSON.stringify(archive,null,2)+'\n',target=path.join(output,variant+'.bundle.json');
    await fs.writeFile(target,bytes,{flag:'wx'});
    console.log(JSON.stringify({variant,archive:target,sha256:sha(bytes),...archive.sourceArchive}));
  }finally{runtime.dispose();}
}
