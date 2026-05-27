from __future__ import annotations

from pathlib import Path

import pandas as pd


columns: set[str] | None = None
dfs: list[pd.DataFrame] = []

# df = pd.read_csv('swe_m_manual_annotation - For Tan.csv')
# print(df['Unnamed: 7'].tolist())

for path in Path.cwd().glob('*.csv'):
    if path.name == 'blackboard.csv':
        continue

    if 'Tan' in path.name:
        name = 'tan'
    elif 'Daniel' in path.name:
        name = 'daniel'
    elif 'Yaseen' in path.name:
        name = 'yaseen'
    else:
        raise Exception(f'unknown name in path {path}')

    df = pd.read_csv(path)
    df = df.dropna(axis=1, how='all')
    if 'Unnamed: 7' in df.columns:
        df = df.drop(columns='Unnamed: 7')
    df = df.rename(columns={
        'instance_id (PR_ID)': 'instance_id',
        'issue_link (we need to find!)': 'issue_link',
        'problem_statement': 'problem_statement',
        'issue_cat': 'issue_category',
        'image_assets': 'image_assets',
        'img_cat_1': 'image_category_1',
        'img_cat_2': 'image_category_2',
    })
    df['name'] = name
    df = df.copy()

    print(f'{path.name} ({name}) has {len(df)} rows')

    if columns is None:
        columns = set(df.columns)
    else:
        if set(df.columns) != columns:
            print(columns)
            print(set(df.columns))
            raise Exception('columns not equal')
    dfs.append(df)

full_df = pd.concat(dfs)
full_df.to_csv('blackboard.csv', index=False)
