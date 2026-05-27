from __future__ import annotations

import fcntl
import os
import tempfile
from dataclasses import dataclass, fields
from typing import Optional

import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu


CSV_PATH = 'running_log.csv'
FIELDNAMES = ['name', 'instance_id', 'issue_link', 'problem_statement', 'image_assets', 'key', 'value']

@dataclass
class AnnotationRow:
    name: str
    instance_id: str
    issue_link: str
    problem_statement: str
    image_assets: str
    key: str
    value: str

    def to_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}


def append_annotation(row: AnnotationRow, path: str = CSV_PATH) -> None:
    dir_ = os.path.dirname(os.path.abspath(path))
    new_row = pd.DataFrame([row.to_dict()])

    with open(path, 'a', newline='', encoding='utf-8') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            existing = pd.read_csv(path) if os.path.getsize(path) > 0 else pd.DataFrame(columns=FIELDNAMES)
            updated = pd.concat([existing, new_row], ignore_index=True)

            with tempfile.NamedTemporaryFile(mode='w', dir=dir_, delete=False, suffix='.tmp') as tmp:
                tmp_path = tmp.name
                updated.to_csv(tmp, index=False)

            os.replace(tmp_path, path)
        except Exception:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


KNOWN_USERS = ['mahnsi', 'tan', 'daniel', 'yaseen', 'kumar', 'maleknaz']


class UserNotInStateException(Exception): pass


class SessionState:

    session_state: Optional[SessionState] = None

    @staticmethod
    def get_session_state() -> SessionState:
        if SessionState.session_state is None:
            SessionState.session_state = SessionState()
        return SessionState.session_state

    def __getitem__(self, key: str) -> Optional[str]:
        if hasattr(st.session_state, key):
            return getattr(st.session_state, key)

    def __setitem__(self, key, value):
        setattr(st.session_state, key, value)

    def __delitem__(self, key):
        if hasattr(st.session_state, key):
            delattr(st.session_state, key)

    def __contains__(self, key: str) -> bool:
        return hasattr(st.session_state, key)

    @property
    def user(self) -> str:
        if 'user' in self:
            return self['user']
        raise UserNotInStateException('user not in state')

    @user.setter
    def user(self, value: str) -> None:
        self['user'] = value

    @user.deleter
    def user(self) -> None:
        del self['user']


session_state = SessionState.get_session_state()

REQUIRED_KEYS = {'issue_category', 'image_category_1', 'image_category_2',}
STATUS_ICON = {'complete': '🟢', 'partial': '🟡', 'none': '🔴'}

def get_instance_status(df: pd.DataFrame, instance_id: str, image_assets: list, user: str) -> str:
    user_rows = df[
        (df['instance_id'] == instance_id) &
        (df['name'] == user)
    ]
    if user_rows.empty:
        return 'none'
    total_required = len(image_assets) * len(REQUIRED_KEYS)
    completed = sum(
        len(set(user_rows[user_rows['image_assets'] == asset]['key'].unique()) & REQUIRED_KEYS)
        for asset in image_assets
    )
    return 'complete' if completed >= total_required else 'partial'


def build_sidebar_labels(df_full: pd.DataFrame, df_annotations: pd.DataFrame, user: str) -> dict[str, str]:
    """Returns a mapping of display label -> instance_id."""
    labels = {}
    for iid in sorted(df_full['instance_id'].unique()):
        assets = df_full[df_full['instance_id'] == iid]['image_assets'].unique().tolist()
        status = get_instance_status(df_annotations, iid, assets, user)
        labels[f"{STATUS_ICON[status]} {iid}"] = iid
    return labels


def get_existing_value(df: pd.DataFrame, user: str, instance_id: str, key: str) -> str | None:
    row = df[
        (df['name'] == user) &
        (df['instance_id'] == instance_id) &
        (df['key'] == key)
    ]
    return row.iloc[-1]['value'] if not row.empty else None


ISSUE_CATEGORIES = [
    '1.1 Incomplete data processing',
    '1.2 Missing input validation',
    '1.2.3 Missing null check',
    '1.2.5 Missing handling of special characters',
    '1.3 Error handling',
    '1.4 Incomplete configuration processing',
    '2 Incorrect feature impl.',
    '2.1 Incorrect data processing ',
    '2.1.2 Incorrect initialization',
    '2.2 Incorrect input validation',
    '2.2.2 Incorrect handling of special characters',
    '2.4 Incorrect output',
    '2.4.1 Incorrect output message',
    '2.5 Incorrect configuration processing',
    '2.7 Performance',
    '4 Perfective maintenance',
]

CAT1_OPTIONS = [
    "Code Snippet Screenshot",
    "Web Interface (UI/UX Element)",
    'Map/Geospatial Visualization',
    'Diagram',
    'Data Visualization',
    'Artwork / Photography',
    'Error Message',
    'Miscellaneous'
]

CAT2_OPTIONS = [
    "Code",
    "Run Time Error",
    'Menus and Preference',
    'Dialog Box',
    'Steps and Processes',
    'Program Input',
    'Desired Output',
    'Program Output',
    'CPU/GPU Performance',
    'Algorithm/Concept Description'
]


def login_screen():
    st.title('Who are you?')
    name = st.text_input('Enter your name')
    if st.button('Continue'):
        if name.strip().lower() in KNOWN_USERS:
            session_state.user = name
            st.rerun()
        else:
            st.error('Name not recognized.')


def home_screen():
    st.title(f'Welcome, {session_state.user}')
    st.write('This is the home page.')

    df_full = pd.read_csv(CSV_PATH)
    labels = build_sidebar_labels(df_full, df_full, session_state.user)

    with st.sidebar:
        st.title("Navigation")
        selected_label = option_menu(
            "Menu",
            list(labels.keys()),
            icons=None,
            styles={
                "nav-link": {"font-size": "12px", "padding": "4px 8px"},
                "nav-link-selected": {"font-size": "12px"},
                "icon": {"display": "none"},
            },
        )

    instance_id = labels[selected_label]
    df = df_full.query(f'instance_id == "{instance_id}"').copy()
    image_assets = df['image_assets'].unique().tolist()

    for image_asset in image_assets:

        if pd.isna(image_asset) or not image_asset:
            st.warning("No image available for this instance.")
            continue

        st.image(image_asset, use_container_width=True)

        # Before the form:
        existing_issue_cat = get_existing_value(df_full, session_state.user, instance_id, 'issue_category')
        default_issue_cat = ISSUE_CATEGORIES.index(existing_issue_cat) if existing_issue_cat in ISSUE_CATEGORIES else 0

        with st.form("issue_category"):
            st.subheader("Issue Category")
            if existing_issue_cat:
                st.info(f"Previously submitted: {existing_issue_cat}")
            issue_category = st.selectbox(
                "Issue Category",
                ISSUE_CATEGORIES,
                index=default_issue_cat,
            )
            submitted = st.form_submit_button("Submit")
            if submitted:
                append_annotation(AnnotationRow(
                    name=session_state.user,
                    instance_id=instance_id,
                    issue_link=...,
                    problem_statement=...,
                    image_assets=...,
                    key='image_category_2',
                    value=issue_category
                ))
                st.success("Saved.")

        existing_cat1 = get_existing_value(df_full, session_state.user, instance_id, 'image_category_1')
        default_cat1 = CAT1_OPTIONS.index(existing_cat1) if existing_cat1 in CAT1_OPTIONS else 0
        with st.form("category_1"):
            st.subheader("Category 1")
            if existing_cat1:
                st.info(f"Previously submitted: {existing_cat1}")
            cat1 = st.selectbox("Image cat 1", CAT1_OPTIONS, index=default_cat1)
            submitted = st.form_submit_button("Submit")
            if submitted:
                append_annotation(AnnotationRow(
                    name=session_state.user,
                    instance_id=instance_id,
                    issue_link=...,
                    problem_statement=...,
                    image_assets=...,
                    key='image_category_1',
                    value=cat1
                ))
                st.success("Saved.")

        existing_cat2 = get_existing_value(df_full, session_state.user, instance_id, 'image_category_2')
        default_cat2 = CAT2_OPTIONS.index(existing_cat2) if existing_cat2 in CAT2_OPTIONS else 0
        with st.form("category_2"):
            st.subheader("Category 2")
            if existing_cat2:
                st.info(f"Previously submitted: {existing_cat2}")
            cat2 = st.selectbox("Image cat 2", CAT2_OPTIONS, index=default_cat2)
            submitted = st.form_submit_button("Submit")
            if submitted:
                append_annotation(AnnotationRow(
                    name=session_state.user,
                    instance_id=instance_id,
                    issue_link=...,
                    problem_statement=...,
                    image_assets=...,
                    key='image_category_2',
                    value=cat2
                ))
                st.success("Saved.")

        with st.form("image_quality"):
            st.selectbox("Rating", ["Excellent", "Good", "Fair", "Poor"])
            st.form_submit_button("Submit")

    if st.button("Log out"):
        del session_state.user
        st.rerun()


if 'user' not in session_state:
    login_screen()
else:
    st.caption(f'Current user: {session_state.user}')
    home_screen()
