"""Training-only profiles; no experiment archive or task label is required."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype


def json_value(value):
    """将单个表格值转成 JSON 可保存的值；无穷值保留为文字标记。"""
    if pd.isna(value):
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return "Infinity" if value > 0 else "-Infinity"
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def numeric_column(series):
    # 布尔值按类别描述，避免把 True/False 当作连续测量。
    return is_numeric_dtype(series.dtype) and not is_bool_dtype(series.dtype)


def correlation_ratio(values, groups):
    """相关比 eta：类别之间的数值均值差异占总变异的比例，范围0–1。"""
    if len(values) < 2 or groups.nunique() < 2:
        return None
    overall = values.mean()
    total = ((values - overall) ** 2).sum()
    if total == 0:
        return None
    frame = pd.DataFrame({'value': values, 'group': groups})
    grouped = frame.groupby('group', observed=True)['value'].agg(['size', 'mean'])
    between = (grouped['size'] * (grouped['mean'] - overall) ** 2).sum()
    return float(np.sqrt(np.clip(between / total, 0, 1)))


def target_association(series, target, task):
    """只用特征与标签均非缺失、数值有限的配对行计算关联。"""
    numeric = numeric_column(series)
    mask = series.notna() & target.notna()
    if numeric:
        mask &= np.isfinite(series)
    if task == 'regression':
        mask &= np.isfinite(target)
    elif numeric_column(target):
        mask &= np.isfinite(target)
    # 去除原索引后按位置配对，避免重复行索引影响统计。
    x, y = series[mask].reset_index(drop=True), target[mask].reset_index(drop=True)
    value = None
    if task == 'regression' and numeric:
        metric = 'pearson_r'
        if len(x) >= 2 and x.nunique() > 1 and y.nunique() > 1:
            value = float(x.astype(float).corr(y.astype(float)))
    elif numeric or task == 'regression':
        metric = 'correlation_ratio_eta'
        values, groups = (x.astype(float), y) if numeric else (y.astype(float), x)
        value = correlation_ratio(values, groups)
    else:
        metric = 'cramers_v'
        table = pd.crosstab(x, y).to_numpy(dtype=float)
        if len(x) and min(table.shape) > 1:
            expected = np.outer(table.sum(axis=1), table.sum(axis=0)) / len(x)
            chi_square = (((table - expected) ** 2) / expected).sum()
            value = float(np.sqrt(np.clip(chi_square / (len(x) * (min(table.shape) - 1)), 0, 1)))
    return {'metric': metric, 'value': value, 'paired_rows': len(x)}


def profile_table(train, target=None, task='classification', top_k=5):
    """train只包含训练折的特征；target是同索引、同行顺序的可选标签Series。"""
    if task not in ('classification', 'regression') or top_k < 1:
        raise ValueError('task须为classification或regression，top_k须大于0')
    if not train.columns.is_unique or not all(isinstance(c, str) for c in train.columns):
        raise ValueError('特征列名须为不重复的字符串')
    if target is not None:
        if not isinstance(target, pd.Series) or not target.index.equals(train.index):
            raise ValueError('target须为与train索引和行顺序一致的Series')
        if task == 'regression':
            target = pd.to_numeric(target, errors='raise')
    rows = len(train)
    profiles = []
    for name, series in train.items():
        missing = int(series.isna().sum())
        numeric = numeric_column(series)
        column = {'name': name, 'dtype': str(series.dtype),
                  'kind': 'numeric' if numeric else 'categorical',
                  'missing_count': missing, 'missing_rate': missing / rows if rows else None,
                  'unique_count': int(series.nunique(dropna=True)),
                  'top_values': [{'value': json_value(v), 'count': int(n),
                                  'rate': int(n) / rows}
                                 for v, n in series.value_counts(dropna=True).head(top_k).items()
                                 if n > 0]}
        if numeric:
            # NaN计为缺失；正负无穷单独计数，不参与分位数或IQR统计。
            observed = series.dropna().astype(float)
            finite = observed[np.isfinite(observed)]
            column['finite_count'] = len(finite)
            column['infinite_count'] = len(observed) - len(finite)
            quantiles = finite.quantile([0, .05, .25, .5, .75, .95, 1])
            column['quantiles'] = {key: json_value(v) for key, v in
                                   zip(['min', 'p05', 'q25', 'median', 'q75', 'p95', 'max'], quantiles)}
            q1, q3 = quantiles.loc[.25], quantiles.loc[.75]
            lower, upper = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
            count = int(((finite < lower) | (finite > upper)).sum())
            column['iqr_outliers'] = {'lower_bound': json_value(lower), 'upper_bound': json_value(upper),
                                      'count': count, 'rate': count / len(finite) if len(finite) else None}
        if target is not None:
            column['target_association'] = target_association(series, target, task)
        profiles.append(column)
    result = {'rows': rows, 'feature_count': train.shape[1],
              'configuration': {'top_k': top_k, 'sample_rows': 5, 'iqr_multiplier': 1.5,
                                'quantile_interpolation': 'linear'},
              'column_profiles': profiles,
              'sample_rows': [{c: json_value(v) for c, v in row.items()}
                              for row in train.head(5).to_dict(orient='records')]}
    if target is not None:
        result['target'] = {'name': json_value(target.name), 'task': task,
                            'missing_count': int(target.isna().sum())}
        result['sample_targets'] = [json_value(v) for v in target.head(5)]
    return result


def profile_text(profile):
    """每列一段短文本，供人工阅读或后续模型提示使用。"""
    lines = [f"训练数据：{profile['rows']}行，{profile['feature_count']}个特征。"]
    if 'target' in profile:
        lines.append('目标：' + json.dumps(profile['target'], ensure_ascii=False))
    for column in profile['column_profiles']:
        rate = column['missing_rate']
        missing = '未定义' if rate is None else f'{rate:.2%}'
        line = f"{column['name']}：类型={column['dtype']}；缺失率={missing}；唯一值数={column['unique_count']}"
        line += '；高频值=' + json.dumps(column['top_values'], ensure_ascii=False)
        if column['kind'] == 'numeric':
            line += '；分位数=' + json.dumps(column['quantiles'], ensure_ascii=False)
            line += '；IQR极端值=' + json.dumps(column['iqr_outliers'], ensure_ascii=False)
            line += f"；无穷值数={column['infinite_count']}"
        if 'target_association' in column:
            line += '；目标关联=' + json.dumps(column['target_association'], ensure_ascii=False)
        lines.append(line)
    lines.append('前5行样例：' + json.dumps(profile['sample_rows'], ensure_ascii=False))
    if 'sample_targets' in profile:
        lines.append('对应标签：' + json.dumps(profile['sample_targets'], ensure_ascii=False))
    return '\n'.join(lines) + '\n'
