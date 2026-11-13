"""Actual per-process CPU/RSS/wall measurements, without inferred historic values."""
from contextlib import contextmanager
from pathlib import Path
import os,time,threading,resource,json,shutil
import psutil
@contextmanager
def measured(path):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);start=time.monotonic();u=resource.getrusage(resource.RUSAGE_SELF);stop=threading.Event();samples=[];proc=psutil.Process();disk0=shutil.disk_usage(path.parent).free
 def observe():
  while not stop.wait(.25):
   try:samples.append(proc.memory_info().rss)
   except psutil.Error:break
 thread=threading.Thread(target=observe,daemon=True);thread.start();state='COMPLETED'
 try:yield
 except BaseException:state='FAILED_OR_INTERRUPTED';raise
 finally:
  stop.set();thread.join();v=resource.getrusage(resource.RUSAGE_SELF);rss=proc.memory_info().rss
  result={'pid':os.getpid(),'status':state,'wall_seconds':time.monotonic()-start,'CPU_user_seconds':v.ru_utime-u.ru_utime,'CPU_system_seconds':v.ru_stime-u.ru_stime,'RSS_sampled_peak_bytes':max(samples+[rss]),'RSS_sample_interval_seconds':.25,'process_lifetime_ru_maxrss_KiB':v.ru_maxrss,'free_disk_before_bytes':disk0,'free_disk_after_bytes':shutil.disk_usage(path.parent).free,'GPU_training_seconds':0,'scope':'this CPU process; thread CPU included; CUDA runtime not used'};temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(result,indent=2));os.replace(temp,path)
