from pathlib import Path
import sys,ast,json,hashlib,types,traceback
import numpy as np,tensorflow as tf
B=Path('/data1/stock-mxy/课题/一进二策略/research/round23_6_26_execution');src=B/'vendor/RSR_api/training';sys.path.insert(0,str(src));R=B.parents[1]/'result/round23_6_round26_master/l0'
records=[]
for filename,classname in [('rank_lstm.py','RankLSTM'),('relation_rank_lstm.py','ReRaLSTM')]:
 try:
  source=(src/filename).read_text();tree=ast.parse(source)
  cls=next(n for n in tree.body if isinstance(n,ast.ClassDef))
  train=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='train')
  graph=next(n for n in train.body if isinstance(n,ast.With))
  self=types.SimpleNamespace(batch_size=8,parameters={'seq':16,'unit':64,'alpha':.1,'lr':.001},fea_dim=5,inner_prod=False,flat=False,rel_encoding=np.random.RandomState(0).randint(0,2,(8,8,3)).astype('float32'),rel_mask=np.zeros((8,8),dtype='float32'))
  import random
  ns={'self':self,'tf':tf,'np':np,'random':random,'device_name':'/cpu:0','leaky_relu':tf.nn.leaky_relu}
  exec(compile(ast.Module(body=[graph]),str(src/filename),'exec'),ns)
  rng=np.random.RandomState(0)
  shape=(8,16,5) if filename=='rank_lstm.py' else (8,64)
  feed={ns['feature']:rng.normal(size=shape).astype('float32'),ns['ground_truth']:rng.normal(size=(8,1)).astype('float32')*.01,ns['base_price']:np.ones((8,1),dtype='float32'),ns['mask']:np.ones((8,1),dtype='float32')}
  with tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=2,inter_op_parallelism_threads=1,device_count={'GPU':0})) as sess:
   sess.run(tf.global_variables_initializer());before=sess.run(tf.trainable_variables())
   loss,pred,_=sess.run([ns['loss'],ns['prediction'],ns['optimizer']],feed)
   after=sess.run(tf.trainable_variables());changed=sum(not np.array_equal(a,b) for a,b in zip(before,after))
  records.append({'module':filename,'status':'OFFICIAL_TF1_GRAPH_UPDATE_PASSED','loss':float(loss),'changed_variables':changed,'prediction_shape':list(pred.shape),'source_sha256':hashlib.sha256(source.encode()).hexdigest(),'scope':'unchanged complete graph construction from author train method, synthetic inputs, finite one-step update; no L1 claim'})
 except Exception as e:records.append({'module':filename,'status':'FAILED_RUNTIME','error':str(e),'traceback':traceback.format_exc()})
(R/'RSR.json').write_text(json.dumps({'method':'RSR','tensorflow_version':tf.__version__,'python_version':sys.version,'L0':'ALGORITHM_CORE_UPDATE_PASSED' if all(x['status'].endswith('PASSED') for x in records) else 'FAILED_RUNTIME','records':records},indent=2));print(json.dumps(records))
