#!/usr/bin/env python3
"""Compile adapter against official ns-3.44 wheel libraries (Linux)."""
import argparse,hashlib,json,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--ns3-root',required=True,type=Path);a=p.parse_args()
r=a.ns3_root.resolve();here=Path(__file__).resolve().parent
libs=r/'lib64';source=here/'ns3_bridge.cc';out=here/'ns3_bridge.so'
command=['g++','-std=c++20','-O2','-shared','-fPIC',str(source),'-I'+str(r/'include'),'-L'+str(libs),'-Wl,-rpath,'+str(libs),'-lns3.44-propagation','-lns3.44-mobility','-lns3.44-network','-lns3.44-core','-o',str(out)]
subprocess.run(command,check=True)
meta={'ns3_version':'3.44.post0','adapter_source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'propagation_library_sha256':hashlib.sha256((libs/'libns3.44-propagation.so').read_bytes()).hexdigest(),'compiler':subprocess.check_output(['g++','--version'],text=True).splitlines()[0]}
(here.parent/'vcc_results/ns3_build.json').write_text(json.dumps(meta,indent=2));print(json.dumps(meta,indent=2))
