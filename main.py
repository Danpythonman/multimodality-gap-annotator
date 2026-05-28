from __future__ import annotations

import fcntl
import os
import json
import requests
import tempfile
from dataclasses import dataclass, fields
from pathlib import Path
from io import BytesIO
from typing import Optional

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from streamlit_option_menu import option_menu
from streamlit_drawable_canvas import st_canvas
from PIL import Image, ImageDraw

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


KNOWN_USERS = ['mahnsi', 'tan', 'daniel', 'yaseen', 'kumar', 'parsa', 'maleknaz']


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
    if completed == 0:
        return 'none'
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


def get_next_incomplete(df: pd.DataFrame, user: str, current_instance_id: str) -> str | None:
    all_ids = sorted(df['instance_id'].unique().tolist())
    current_index = all_ids.index(current_instance_id)
    # Search from current position forward, then wrap around
    search_order = all_ids[current_index + 1:] + all_ids[:current_index]
    for iid in search_order:
        assets = df[df['instance_id'] == iid]['image_assets'].unique().tolist()
        if get_instance_status(df, iid, assets, user) != 'complete':
            return iid
    return None


def get_previous_incomplete(df: pd.DataFrame, user: str, current_instance_id: str) -> str | None:
    all_ids = sorted(df['instance_id'].unique().tolist())
    current_index = all_ids.index(current_instance_id)
    search_order = all_ids[:current_index][::-1] + all_ids[current_index + 1:][::-1]
    for iid in search_order:
        assets = df[df['instance_id'] == iid]['image_assets'].unique().tolist()
        if get_instance_status(df, iid, assets, user) != 'complete':
            return iid
    return None


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


def draw_box(img, box):
    overlay = img.copy()
    draw = ImageDraw.Draw(overlay)
    x, y, w, h = box["x"], box["y"], box["w"], box["h"]
    draw.rectangle([x, y, x + w, y + h], outline="red", width=2)
    return overlay


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
    if not session_state.user:
        raise Exception('user not logged in!')

    df_full = pd.read_csv(CSV_PATH)
    df_full = df_full.query(f'name == "{session_state.user}"').copy()

    if 'selected_instance' not in session_state:
        session_state['selected_instance'] = sorted(df_full['instance_id'].unique())[0]

    labels = build_sidebar_labels(df_full, df_full, session_state.user)

    current_label = next((k for k, v in labels.items() if v == session_state['selected_instance']), list(labels.keys())[0])
    default_index = list(labels.keys()).index(current_label)

    with st.sidebar:
        st.title("Navigation")
        selected_label = option_menu(
            "Menu",
            list(labels.keys()),
            icons=None,
            default_index=default_index,
            styles={
                "nav-link": {"font-size": "12px", "padding": "4px 8px"},
                "nav-link-selected": {"font-size": "12px"},
                "icon": {"display": "none"},
            },
        )
        session_state['selected_instance'] = labels[selected_label]

    instance_id = session_state['selected_instance']
    df = df_full.query(f'instance_id == "{instance_id}"').copy()
    image_assets = df['image_assets'].unique().tolist()
    issue_link = df.iloc[0]['issue_link']
    problem_statement = df.iloc[0]['problem_statement']

    with st.container(border=True):
        st.caption(f"Current user: {session_state.user}")
        st.header(instance_id)
        st.markdown(f"🔗 [View Issue]({issue_link})")
        st.text_area("Problem Statement", value=problem_statement, disabled=True, height=300)
    st.divider()

    for i, image_asset in enumerate(image_assets):
        with st.container(border=True):
            if pd.isna(image_asset) or not image_asset:
                st.warning("No image available for this instance.")
                continue

            row = df.query(f'image_assets == "{image_asset}"').iloc[0]

            st.subheader(f'Image {i+1}/{len(image_assets)}')
            st.markdown(f"🔗 [View Image]({image_asset})")

            st.image(image_asset, width='stretch')

            existing_issue_cat = get_existing_value(df_full, session_state.user, instance_id, 'issue_category')
            default_issue_cat = ISSUE_CATEGORIES.index(existing_issue_cat) if existing_issue_cat in ISSUE_CATEGORIES else 0

            with st.form(f"issue_category-{i}"):
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
                        issue_link=row['issue_link'],
                        problem_statement=row['problem_statement'],
                        image_assets=image_asset,
                        key='issue_category',
                        value=issue_category
                    ))
                    st.success("Saved.")

            existing_cat1 = get_existing_value(df_full, session_state.user, instance_id, 'image_category_1')
            default_cat1 = CAT1_OPTIONS.index(existing_cat1) if existing_cat1 in CAT1_OPTIONS else 0
            with st.form(f"category_1-{i}"):
                st.subheader("Category 1")
                if existing_cat1:
                    st.info(f"Previously submitted: {existing_cat1}")
                cat1 = st.selectbox("Image cat 1", CAT1_OPTIONS, index=default_cat1)
                submitted = st.form_submit_button("Submit")
                if submitted:
                    append_annotation(AnnotationRow(
                        name=session_state.user,
                        instance_id=instance_id,
                        issue_link=row['issue_link'],
                        problem_statement=row['problem_statement'],
                        image_assets=image_asset,
                        key='image_category_1',
                        value=cat1
                    ))
                    st.success("Saved.")

            existing_cat2 = get_existing_value(df_full, session_state.user, instance_id, 'image_category_2')
            default_cat2 = CAT2_OPTIONS.index(existing_cat2) if existing_cat2 in CAT2_OPTIONS else 0
            with st.form(f"category_2-{i}"):
                st.subheader("Category 2")
                if existing_cat2:
                    st.info(f"Previously submitted: {existing_cat2}")
                cat2 = st.selectbox("Image cat 2", CAT2_OPTIONS, index=default_cat2)
                submitted = st.form_submit_button("Submit")
                if submitted:
                    append_annotation(AnnotationRow(
                        name=session_state.user,
                        instance_id=instance_id,
                        issue_link=row['issue_link'],
                        problem_statement=row['problem_statement'],
                        image_assets=image_asset,
                        key='image_category_2',
                        value=cat2
                    ))
                    st.success("Saved.")

            with st.form(f"image_quality-{i}"):
                st.selectbox("Rating", ["Excellent", "Good", "Fair", "Poor"])
                st.form_submit_button("Submit")

            response = requests.get(image_asset)
            img = Image.open(BytesIO(response.content))

            display_width = 700
            scale = display_width / img.width
            display_height = int(img.height * scale)
            img_resized = img.resize((display_width, display_height))

            canvas_result = st_canvas(
                background_image=img_resized,
                drawing_mode="rect",
                stroke_width=2,
                stroke_color="#ff0000",
                fill_color="rgba(255, 0, 0, 0.1)",
                height=display_height,
                width=display_width,
                key=f"canvas-{i}",
            )

            if canvas_result.json_data:
                rects = [o for o in canvas_result.json_data["objects"] if o["type"] == "rect"]
                if rects:
                    last = rects[-1]
                    box = {
                        "x": int(last["left"] / scale),
                        "y": int(last["top"] / scale),
                        "w": int(last["width"] / scale),
                        "h": int(last["height"] / scale),
                    }
                    if st.button("Save bounding box", key=f"save_bbox_{i}"):
                        append_annotation(AnnotationRow(
                            name=session_state.user,
                            instance_id=instance_id,
                            issue_link=row['issue_link'],
                            problem_statement=row['problem_statement'],
                            image_assets=image_asset,
                            key='bounding_box',
                            value=json.dumps(box)  # e.g. '{"x": 10, "y": 20, "w": 100, "h": 50}'
                        ))
                        st.success(f"Saved: {box}")

        st.divider()

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Log out", width='stretch'):
            del session_state.user
            st.rerun()
    prev_id = get_previous_incomplete(df_full, session_state.user, instance_id)
    with col2:
        if prev_id:
            if st.button("← Previous incomplete", width='stretch'):
                session_state['selected_instance'] = prev_id
                st.rerun()
    next_id = get_next_incomplete(df_full, session_state.user, instance_id)
    with col3:
        if next_id:
            if st.button("Next incomplete →", width='stretch'):
                session_state['selected_instance'] = next_id
                st.rerun()

if 'user' not in session_state:
    login_screen()
else:
    home_screen()
