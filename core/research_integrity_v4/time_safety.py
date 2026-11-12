"""Explicit maturity and revision gates. Missing evidence is never a PASS."""
import ast
import pandas as pd


def mature_training(frame,begin,decision):
    if 'label_available_at' not in frame:raise ValueError('label maturity evidence missing')
    t=pd.to_datetime(frame.trade_date);known=pd.to_datetime(frame.label_available_at)
    return frame.loc[t.ge(pd.Timestamp(begin)) & t.lt(pd.Timestamp(decision)) & known.notna() & known.lt(pd.Timestamp(decision))].copy()


def asof_versions(frame,decision,keys):
    required=['event_time','known_at','revision_time']
    if any(c not in frame for c in required):raise ValueError('revision evidence missing')
    data=frame.copy();cut=pd.Timestamp(decision)
    for c in required:data[c]=pd.to_datetime(data[c])
    data=data.loc[data.event_time.le(cut)&data.known_at.le(cut)&data.revision_time.le(cut)]
    return data.sort_values(keys+['known_at','revision_time'],kind='mergesort').drop_duplicates(keys,keep='last')


def validate_formula(expression,max_nodes=12):
    # Parse only. No eval/exec is exposed to a generated expression.
    features={'open','high','low','close','pre_close','volume','amount','vwap'}
    operators={'Abs','Log','Add','Sub','Mul','Div','Greater','Less','Ref','Mean','Sum','Std','Var','Max','Min','Med','Mad','Delta','WMA','EMA','Cov','Corr'}
    source=expression.replace('$','');tree=ast.parse(source,mode='eval')
    calls=[x for x in ast.walk(tree) if isinstance(x,ast.Call)]
    if len(calls)>max_nodes:raise ValueError('expression budget exceeded')
    for node in ast.walk(tree):
        if isinstance(node,ast.Call):
            if not isinstance(node.func,ast.Name) or node.func.id not in operators:raise ValueError('unsupported operator')
            if node.keywords:raise ValueError('keyword operations not allowed')
            if node.func.id in {'Ref','Mean','Sum','Std','Var','Max','Min','Med','Mad','Delta','WMA','EMA','Cov','Corr'}:
                window=node.args[-1]
                if not isinstance(window,ast.Constant) or window.value not in {1,5,10,20,60}:raise ValueError('invalid/forward window')
        elif isinstance(node,ast.Name) and node.id not in features|operators:raise ValueError('forbidden feature')
        elif isinstance(node,(ast.Attribute,ast.Subscript,ast.Lambda,ast.ListComp,ast.DictComp)):raise ValueError('unsupported language construct')
    return ast.dump(tree,annotate_fields=True,include_attributes=False)
