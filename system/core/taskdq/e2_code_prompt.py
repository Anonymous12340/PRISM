"""Explicit fit/apply prompt adapter; frozen E2 pilot keeps its original prompts."""
import json
import re
from . import e2_operations as ops


CAAFE_FORMAT_MARKER='\nCode formatting for each added column:'
SIGNATURE='def modify_features(df_input, state=None, fit=False):'


def caafe_context(original,task_type):
    """Keep upstream semantic instructions; replace its incompatible output ABI."""
    if CAAFE_FORMAT_MARKER not in original:
        raise ValueError('Upstream CAAFE formatting section changed; inspect before adapting')
    context=original.split(CAAFE_FORMAT_MARKER,1)[0]
    context=context.replace('categoricals encoded as int','categoricals are object strings')
    if task_type=='classification':
        context=context.replace('The evaluation metric is accuracy.','The evaluation metric is AUC (higher is better).')
    elif task_type=='regression':
        context=context.replace('classification algorithm','regression algorithm').replace('downstream classifier','downstream regressor')
        context=context.replace('The classifier will be trained','The regressor will be trained')
        context=context.replace('The evaluation metric is accuracy.','The evaluation metric is RMSE (lower is better).')
    else:raise ValueError('Unsupported task type')
    return (context+'\nPropose exactly one useful additional feature in the CURRENT columns; '
        'redundant original columns may be dropped. df_input contains those current columns. '
        'Return a complete function using df_input/df_output, not a top-level df snippet. '
        'Do not preprocess every existing field; the common scorer already handles its encoding and missing values.')


def llmfe_context(original,compact=False):
    """Keep selected native evolution examples, replace only the unfinished header."""
    headers=list(re.finditer(r'(?m)^def modify_features_v\d+\(',original))
    if not headers:raise ValueError('No native versioned function header found')
    context=original[:headers[-1].start()].rstrip()
    if compact:
        context=context.replace('Please generate as many new features as possible','Please generate at most three new features')
        context=context.replace('Please generate as many new features strictly using','Please generate at most three new features strictly using')
    return (context+
        '\n\nThe versioned functions above are evolution examples, not callable helpers. '
        'Return ONE complete, self-contained improved function named modify_features. '
        'Do not return the examples, the next versioned header or a partial function body. '
        'All selected feature computation must fit in the existing output budget; omit repeated comments and descriptions.')


def messages(method,original,task,feedback,compact=False,optional_statistics=False):
    if method=='caafe':context=caafe_context(original,task['type'])
    elif method=='llmfe':context=llmfe_context(original,compact)
    else:raise ValueError('Only the two generated-code baselines use this adapter')
    system=(ops.CONTRACT+' Response formatting is JSON only, with one code string, no Markdown fences. '
        'The only top-level definition is '+SIGNATURE+' np/pd are already bound; do not import them. '
        'The unchanged output ceiling is 1024 generated tokens, including JSON syntax. '
        'Return the complete closing JSON brace; a truncated response is a failed proposal and is never retried.')
    if compact:
        system+=(' COMPACT MUTATION CONTRACT: the common scorer already handles missing original predictors and encodes categories. '
            'Do NOT loop over or preprocess every input column. Do NOT add constant median/quantile summary columns. '
            'For CAAFE add one next feature; for LLM-FE propose at most three nonconstant features using a small subset of input columns. '
            'Choose any useful operators, not a prescribed action. Fit ONLY the statistics needed by your chosen new features. '
            'Preserve other original columns unchanged, apart from an explicitly justified redundant-column drop. '
            'No repeated prose, imports, comments or long docstring: only concise executable code. '
            'Aim below 40 lines and 700 generated tokens, leaving room for the complete JSON wrapper.')
    if optional_statistics:
        assert method in ['caafe','llmfe']
        old='Learn medians, quantiles, group/category/frequency maps and bin edges ONLY inside an explicit if fit: block.'
        assert old in system
        system=system.replace(old,
            'Pure row-wise arithmetic needs no training statistics. Only if a CHOSEN feature needs a statistic, '
            'compute that statistic inside if fit: and store it in state. Otherwise state may remain empty.')
        feature_limit='exactly one useful nonconstant feature' if method=='caafe' else 'at most three useful nonconstant features'
        context+=('\nEXECUTABLE OUTPUT REMINDER: choose '+feature_limit+' first. '
            'Compute only those chosen features. Training medians/quantiles are optional, not requested outputs. '
            'For a stateless arithmetic proposal use state = {} if state is None else state, '
            'compute your chosen features row-wise, then return (df_output, state). '
            'Return short complete code and close the JSON. All feature choices remain yours.')
    return [{'role':'system','content':system},
        {'role':'user','content':context+'\n\nTASK:\n'+json.dumps(task,ensure_ascii=False)+
            '\n\nTRAINING FEEDBACK:\n'+json.dumps(feedback,ensure_ascii=False)+
            '\n\nOUTPUT CONTRACT: follow the system fit/apply signature and JSON format. '
            'Use no target column as a predictor. Return (df_output, state); fit=False reuses training state.'}]
