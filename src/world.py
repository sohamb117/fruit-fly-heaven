"""Inexhaustible fruit habitat and an explicit, supplied motor decoder.

Coordinates are millimeters. Odor/taste drive measured sensory cells. Motor
rates alone determine locomotion; there is no target-seeking steering rule.
"""
from concurrent.futures import ThreadPoolExecutor
import math
import random
import time
from brain import Brain, Graph

FRUIT = [
    {'kind': 'banana', 'x': -18, 'z': -13, 'angle': -.45, 'length': 38, 'radius': 6, 'y': 11},
    {'kind': 'banana', 'x': -5, 'z': 13, 'angle': .4, 'length': 40, 'radius': 6, 'y': 13},
    {'kind': 'banana', 'x': -12, 'z': 0, 'angle': -.05, 'length': 42, 'radius': 5.7, 'y': 17},
    {'kind': 'apple', 'x': 22, 'z': -15, 'radius': 13, 'y': 13},
    {'kind': 'apple', 'x': 24, 'z': 16, 'radius': 12, 'y': 13},
    {'kind': 'apple', 'x': -30, 'z': 21, 'radius': 10, 'y': 11},
]

def banana_point(f, t):
    x = (t-.5)*f['length']; z = 9*(4*(t-.5)**2-1)
    return (f['x']+math.cos(f['angle'])*x-math.sin(f['angle'])*z,
            f['z']+math.sin(f['angle'])*x+math.cos(f['angle'])*z)

def fruit_distance(f, x, z):
    if f['kind'] == 'apple':
        return math.hypot(x-f['x'], z-f['z'])
    return min(math.hypot(x-p[0], z-p[1]) for p in f['path'])

for f in FRUIT:
    if f['kind']=='banana':
        f['path'] = [banana_point(f, i/30) for i in range(31)]

def habitat(x, z):
    height = 1.5+.0037*(x*x+z*z)
    contact = False
    for f in FRUIT:
        d = fruit_distance(f, x, z)
        r = f['radius']
        if d<r:
            h=f['y']+math.sqrt(max(0,r*r-d*d))
            if h>=height:
                height=h; contact=True
    return height, contact

def odor(x, z):
    return min(1., sum(.3*math.exp(-max(0,fruit_distance(f,x,z)-f['radius'])/18) for f in FRUIT))

class World:
    def __init__(self, count=100, workers=6, seed=20260912):
        self.graph=Graph()
        self.brains=[Brain(self.graph, seed+i*100003) for i in range(count)]
        self.pool=ThreadPoolExecutor(max_workers=workers)
        self.flies=[]; self.paused=False; self.ms=0; self.speed=0
        self.odor_enabled=True; self.taste_enabled=True
        self.started=time.time(); self.wall_compute=0
        rng=random.Random(seed)
        for i in range(count):
            # Put flies on fruit at launch; food seeking remains an emergent question.
            f=FRUIT[i%len(FRUIT)]
            if f['kind']=='apple':
                a=rng.uniform(0,2*math.pi);r=f['radius']*math.sqrt(rng.random())*.9
                x=f['x']+r*math.cos(a);z=f['z']+r*math.sin(a)
            else:
                x,z=banana_point(f,rng.uniform(.07,.93));x+=rng.uniform(-3,3);z+=rng.uniform(-3,3)
            y,contact=habitat(x,z)
            self.flies.append({'id':i+1,'x':x,'y':y,'z':z,'heading':rng.uniform(-math.pi,math.pi),
                               'velocity':0.,'contact':contact,'senses':[0,0,0],
                               'brain':self.brains[i].stats(), 'recent':[]})

    def advance(self):
        start=time.perf_counter()
        stimuli=[]
        for f in self.flies:
            a=f['heading']; x=f['x'];z=f['z']
            # Two antenna samples, 0.8 mm apart. Concentration-to-Hz is assumed.
            l=3+65*odor(x+.9*math.cos(a)-.4*math.sin(a),z+.9*math.sin(a)+.4*math.cos(a)) if self.odor_enabled else 0
            r=3+65*odor(x+.9*math.cos(a)+.4*math.sin(a),z+.9*math.sin(a)-.4*math.cos(a)) if self.odor_enabled else 0
            sweet=150 if f['contact'] and self.taste_enabled else 0
            stimuli.append((l,r,sweet))
        def run(pair):
            b,s=pair
            stats=b.step(200,*s)
            return stats,b.recent()
        outputs=list(self.pool.map(run,zip(self.brains,stimuli)))
        for f,(stats,recent),s in zip(self.flies,outputs,stimuli):
            walk,left,right,feed=[stats[k] for k in ['walk_hz','left_hz','right_hz','feed_hz']]
            # DNa steering cells also participate in walking. This conversion to
            # body speed/turning is deliberately a model, not a measured motor circuit.
            speed=8*math.tanh((walk+.15*(left+right))/30)
            speed/=1+feed/35
            turn=2.8*math.tanh((left-right)/40)
            a=f['heading']+turn*.02
            x=f['x']+math.cos(a)*speed*.02;z=f['z']+math.sin(a)*speed*.02
            radius=math.hypot(x,z)
            if radius>62:
                x*=62/radius;z*=62/radius;speed=0 # bowl collision, no artificial steering
            y,contact=habitat(x,z)
            f.update(x=x,y=y,z=z,heading=a,velocity=speed,contact=contact,
                     senses=s,brain=stats,recent=recent)
        self.ms+=20
        elapsed=time.perf_counter()-start
        self.wall_compute+=elapsed
        self.speed=.02/elapsed

    def snapshot(self,selected=1):
        return {'flies':[{k:v for k,v in f.items() if k!='recent'} for f in self.flies],
                'selected_spikes':self.flies[selected-1]['recent'],
                'time_ms':self.ms,'speed':self.speed,'paused':self.paused,
                'odor_enabled':self.odor_enabled,'taste_enabled':self.taste_enabled,
                'total_spikes':sum(f['brain']['spikes'] for f in self.flies),
                'fruit':FRUIT,'wall_compute_seconds':self.wall_compute}

    def close(self):
        self.pool.shutdown()
        for b in self.brains:b.close()
        self.graph.close()
