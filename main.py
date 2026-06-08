from __future__ import annotations

import fcntl
import json
import os
import tempfile
from dataclasses import dataclass, fields
from io import BytesIO
from typing import Literal, TypedDict, cast

import pandas as pd
import requests
import streamlit as st
from PIL import Image, ImageDraw
from streamlit_drawable_canvas import st_canvas  # pyright: ignore
from streamlit_option_menu import option_menu  # pyright: ignore

CSV_PATH = 'running_log.csv'
FIELDNAMES = [
    'name',
    'instance_id',
    'issue_link',
    'problem_statement',
    'image_assets',
    'key',
    'value',
]


@dataclass(kw_only=True)
class AnnotationRow:
    name: str
    instance_id: str
    issue_link: str
    problem_statement: str
    image_assets: str
    key: str
    value: str

    def to_dict(self) -> dict[str, str]:
        return {f.name: getattr(self, f.name) for f in fields(self)}


def append_annotation(row: AnnotationRow, path: str = CSV_PATH) -> None:
    dir_ = os.path.dirname(os.path.abspath(path))
    new_row = pd.DataFrame([row.to_dict()])

    with open(path, 'a', newline='', encoding='utf-8') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        tmp_path: str | None = None
        try:
            existing = (
                pd.read_csv(path)
                if os.path.getsize(path) > 0
                else pd.DataFrame(columns=FIELDNAMES)
            )
            updated = pd.concat([existing, new_row], ignore_index=True)

            with tempfile.NamedTemporaryFile(
                mode='w', dir=dir_, delete=False, suffix='.tmp'
            ) as tmp:
                tmp_path = tmp.name
                updated.to_csv(tmp, index=False)

            os.replace(tmp_path, path)
        except Exception:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


KNOWN_USERS = [
    'mahnsi',
    'tan',
    'daniel',
    'yaseen',
    'kumar',
    'parsa',
    'maleknaz',
]


class Rect(TypedDict):
    """Streamlit canvas rect."""

    left: float
    top: float
    width: float
    height: float
    type: Literal['rect']


class Box(TypedDict):
    """My saved boxes from the Streamlit canvas"""

    x: float
    y: float
    w: float
    h: float


def draw_boxes(
    img: Image.Image, boxes: list[Box], color: str = 'blue', width: int = 3
) -> Image.Image:
    draw = ImageDraw.Draw(img)
    for box in boxes:
        x0, y0 = box['x'], box['y']
        x1, y1 = x0 + box['w'], y0 + box['h']
        draw.rectangle([x0, y0, x1, y1], outline=color, width=width)
    return img


class UserNotInStateException(Exception):
    pass


class InvalidUserException(Exception):
    pass


class SelectedInstanceNotInStateException(Exception):
    pass


class InvalidSelectedInstanceException(Exception):
    pass


class InvalidCanvasVersionException(Exception):
    pass


type SessionStateValue = str | int | None


class SessionState:
    session_state: SessionState | None = None

    @staticmethod
    def get_session_state() -> SessionState:
        if SessionState.session_state is None:
            SessionState.session_state = SessionState()
        return SessionState.session_state

    def __getitem__(self, key: str) -> SessionStateValue:
        if hasattr(st.session_state, key):
            return getattr(st.session_state, key)
        return None

    def __setitem__(self, key: str, value: SessionStateValue):
        setattr(st.session_state, key, value)

    def __delitem__(self, key: str):
        if hasattr(st.session_state, key):
            delattr(st.session_state, key)

    def __contains__(self, key: str) -> bool:
        return hasattr(st.session_state, key)

    def get(
        self, key: str, default: SessionStateValue = None
    ) -> SessionStateValue:
        return self[key] if key in self else default

    @property
    def user(self) -> str:
        value = self.get('user')
        if not isinstance(value, str):
            raise InvalidUserException('user is not a string')
        if value:
            return value
        raise UserNotInStateException('user not in state')

    @user.setter
    def user(self, value: str) -> None:
        self['user'] = value

    @user.deleter
    def user(self) -> None:
        del self['user']

    @property
    def selected_instance(self) -> str:
        value = self.get('selected_instance')
        if not isinstance(value, str):
            raise InvalidSelectedInstanceException(
                'selected_instance is not a string'
            )
        if value:
            return value
        raise SelectedInstanceNotInStateException(
            'selected_instance not in state'
        )

    @selected_instance.setter
    def selected_instance(self, value: str) -> None:
        self['selected_instance'] = value

    @selected_instance.deleter
    def selected_instance(self) -> None:
        del self['selected_instance']

    @property
    def canvas_version(self) -> int:
        value = self.get('canvas_version')
        if value and not isinstance(value, int):
            raise InvalidCanvasVersionException('canvas_version is not an int')
        if value:
            return value
        self['canvas_version'] = 0
        return 0

    @canvas_version.setter
    def canvas_version(self, value: int) -> None:
        self['canvas_version'] = value

    @canvas_version.deleter
    def canvas_version(self) -> None:
        del self['canvas_version']


session_state = SessionState.get_session_state()

REQUIRED_KEYS = {
    'issue_category',
    'image_category_1',
    'image_category_2',
    'image_quality',
    'bounding_box',
}
STATUS_ICON = {'complete': '🟢', 'partial': '🟡', 'none': '🔴'}


def get_instance_status(
    df: pd.DataFrame, instance_id: str, image_assets: list[str], user: str
) -> str:
    user_rows = df[(df['instance_id'] == instance_id) & (df['name'] == user)]
    if user_rows.empty:
        return 'none'
    total_required = len(image_assets) * len(REQUIRED_KEYS)
    completed = sum(
        len(
            set(user_rows[user_rows['image_assets'] == asset]['key'].unique())
            & REQUIRED_KEYS
        )
        for asset in image_assets
    )
    if completed == 0:
        return 'none'
    return 'complete' if completed >= total_required else 'partial'


def build_sidebar_labels(
    df_full: pd.DataFrame, df_annotations: pd.DataFrame, user: str
) -> dict[str, str]:
    """Returns a mapping of display label -> instance_id."""
    labels: dict[str, str] = {}
    for iid in sorted(cast('pd.Series[str]', df_full['instance_id']).unique()):
        assets = (
            cast(
                'pd.Series[str]',
                df_full[df_full['instance_id'] == iid]['image_assets'],
            )
            .unique()
            .tolist()
        )
        status = get_instance_status(df_annotations, iid, assets, user)
        labels[f'{STATUS_ICON[status]} {iid}'] = iid
    return labels


def get_existing_value(
    df: pd.DataFrame, user: str, instance_id: str, key: str
) -> list[str] | None:
    rows = df[
        (df['name'] == user)
        & (df['instance_id'] == instance_id)
        & (df['key'] == key)
    ]
    return rows['value'].tolist() if not rows.empty else None


def get_next_incomplete(
    df: pd.DataFrame, user: str, current_instance_id: str
) -> str | None:
    all_ids = sorted(df['instance_id'].unique().tolist())
    current_index = all_ids.index(current_instance_id)
    # Search from current position forward, then wrap around
    search_order = all_ids[current_index + 1 :] + all_ids[:current_index]
    for iid in search_order:
        assets = (
            cast(
                'pd.Series[str]', df[df['instance_id'] == iid]['image_assets']
            )
            .unique()
            .tolist()
        )
        if get_instance_status(df, iid, assets, user) != 'complete':
            return iid
    return None


def get_previous_incomplete(
    df: pd.DataFrame, user: str, current_instance_id: str
) -> str | None:
    all_ids = sorted(df['instance_id'].unique().tolist())
    current_index = all_ids.index(current_instance_id)
    search_order = (
        all_ids[:current_index][::-1] + all_ids[current_index + 1 :][::-1]
    )
    for iid in search_order:
        assets = (
            cast(
                'pd.Series[str]', df[df['instance_id'] == iid]['image_assets']
            )
            .unique()
            .tolist()
        )
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
    'Code Snippet Screenshot',
    'Web Interface (UI/UX Element)',
    'Map/Geospatial Visualization',
    'Diagram',
    'Data Visualization',
    'Artwork / Photography',
    'Error Message',
    'Miscellaneous',
]

CAT2_OPTIONS = [
    'Code',
    'Run Time Error',
    'Menus and Preference',
    'Dialog Box',
    'Steps and Processes',
    'Program Input',
    'Desired Output',
    'Program Output',
    'CPU/GPU Performance',
    'Algorithm/Concept Description',
]

IMAGE_QUALITY_OPTIONS = [
    'Not applicable',
    'Not helpful',
    'Somewhat helpful',
    'Highly helpful',
]


@st.cache_data
def fetch_image_bytes(url: str) -> bytes:
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    return response.content


def load_image(url: str) -> Image.Image:
    return Image.open(BytesIO(fetch_image_bytes(url))).convert('RGB')


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
        session_state.selected_instance = sorted(
            df_full['instance_id'].unique()
        )[0]

    labels = build_sidebar_labels(df_full, df_full, session_state.user)

    current_label = next(
        (k for k, v in labels.items() if v == session_state.selected_instance),
        list(labels.keys())[0],
    )
    default_index = list(labels.keys()).index(current_label)

    with st.sidebar:
        st.title('Navigation')
        selected_label = option_menu(
            'Menu',
            list(labels.keys()),
            icons=None,
            default_index=default_index,
            styles={
                'nav-link': {'font-size': '12px', 'padding': '4px 8px'},
                'nav-link-selected': {'font-size': '12px'},
                'icon': {'display': 'none'},
            },
        )
        session_state.selected_instance = labels[selected_label]

    instance_id = session_state.selected_instance
    df = df_full.query(f'instance_id == "{instance_id}"').copy()
    image_assets = df['image_assets'].unique().tolist()
    issue_link = df.iloc[0]['issue_link']
    problem_statement = df.iloc[0]['problem_statement']

    with st.container(border=True):
        st.caption(f'Current user: {session_state.user}')
        st.header(instance_id)
        st.markdown(f'🔗 [View Issue]({issue_link})')
        st.text_area(
            'Problem Statement',
            value=problem_statement,
            disabled=True,
            height=300,
        )
    st.divider()

    for i, image_asset in enumerate(image_assets):
        with st.container(border=True):
            if pd.isna(image_asset) or not image_asset:
                st.warning('No image available for this instance.')
                continue

            row = df.query(f'image_assets == "{image_asset}"').iloc[0]

            st.subheader(f'Image {i + 1}/{len(image_assets)}')
            st.markdown(f'🔗 [View Image]({image_asset})')

            st.image(image_asset, width='stretch')

            existing_issue_cats = get_existing_value(
                df_full, session_state.user, instance_id, 'issue_category'
            )
            default_issue_cat = (
                ISSUE_CATEGORIES.index(existing_issue_cats[-1])
                if existing_issue_cats
                and len(existing_issue_cats) > 0
                and existing_issue_cats[-1] in ISSUE_CATEGORIES
                else 0
            )

            with st.form(f'issue_category-{i}'):
                st.subheader('Issue Category')
                if existing_issue_cats:
                    for existing_issue_cat in existing_issue_cats:
                        st.info(f'Previously submitted: {existing_issue_cat}')
                issue_category = st.selectbox(
                    'Issue Category',
                    ISSUE_CATEGORIES,
                    index=default_issue_cat,
                )
                submitted = st.form_submit_button('Submit')
                if submitted:
                    append_annotation(
                        AnnotationRow(
                            name=session_state.user,
                            instance_id=instance_id,
                            issue_link=row['issue_link'],
                            problem_statement=row['problem_statement'],
                            image_assets=image_asset,
                            key='issue_category',
                            value=issue_category,
                        )
                    )
                    st.success('Saved.')

            existing_cat1s = get_existing_value(
                df_full, session_state.user, instance_id, 'image_category_1'
            )
            default_cat1 = (
                CAT1_OPTIONS.index(existing_cat1s[-1])
                if existing_cat1s
                and len(existing_cat1s) > 0
                and existing_cat1s[-1] in CAT1_OPTIONS
                else 0
            )
            with st.form(f'category_1-{i}'):
                st.subheader('Category 1')
                if existing_cat1s:
                    for existing_cat1 in existing_cat1s:
                        st.info(f'Previously submitted: {existing_cat1}')
                cat1 = st.selectbox(
                    'Image cat 1', CAT1_OPTIONS, index=default_cat1
                )
                submitted = st.form_submit_button('Submit')
                if submitted:
                    append_annotation(
                        AnnotationRow(
                            name=session_state.user,
                            instance_id=instance_id,
                            issue_link=row['issue_link'],
                            problem_statement=row['problem_statement'],
                            image_assets=image_asset,
                            key='image_category_1',
                            value=cat1,
                        )
                    )
                    st.success('Saved.')

            existing_cat2s = get_existing_value(
                df_full, session_state.user, instance_id, 'image_category_2'
            )
            default_cat2 = (
                CAT2_OPTIONS.index(existing_cat2s[-1])
                if existing_cat2s
                and len(existing_cat2s) > 0
                and existing_cat2s[-1] in CAT2_OPTIONS
                else 0
            )
            with st.form(f'category_2-{i}'):
                st.subheader('Category 2')
                if existing_cat2s:
                    for existing_cat2 in existing_cat2s:
                        st.info(f'Previously submitted: {existing_cat2}')
                cat2 = st.selectbox(
                    'Image cat 2', CAT2_OPTIONS, index=default_cat2
                )
                submitted = st.form_submit_button('Submit')
                if submitted:
                    append_annotation(
                        AnnotationRow(
                            name=session_state.user,
                            instance_id=instance_id,
                            issue_link=row['issue_link'],
                            problem_statement=row['problem_statement'],
                            image_assets=image_asset,
                            key='image_category_2',
                            value=cat2,
                        )
                    )
                    st.success('Saved.')

            existing_image_qualitys = get_existing_value(
                df_full, session_state.user, instance_id, 'image_quality'
            )
            default_image_quality = (
                IMAGE_QUALITY_OPTIONS.index(existing_image_qualitys[-1])
                if existing_image_qualitys
                and len(existing_image_qualitys) > 0
                and existing_image_qualitys[-1] in CAT2_OPTIONS
                else 0
            )
            with st.form(f'image_quality-{i}'):
                if existing_image_qualitys:
                    for existing_image_quality in existing_image_qualitys:
                        st.info(
                            f'Previously submitted: {existing_image_quality}'
                        )
                image_quality = st.selectbox(
                    'Image Quality Rating',
                    IMAGE_QUALITY_OPTIONS,
                    index=default_image_quality,
                )
                submitted = st.form_submit_button('Submit')
                if submitted:
                    append_annotation(
                        AnnotationRow(
                            name=session_state.user,
                            instance_id=instance_id,
                            issue_link=row['issue_link'],
                            problem_statement=row['problem_statement'],
                            image_assets=image_asset,
                            key='image_quality',
                            value=image_quality,
                        )
                    )
                    st.success('Saved.')

            with st.container(border=True):
                st.subheader('Important part(s) of image')
                st.write(
                    'Draw a rectangle around the important part(s) of the '
                    'image, then click submit.'
                )

                try:
                    img = load_image(image_asset)
                except Exception as e:
                    st.error(f'Failed to load image: {e}')
                    continue

                display_width = min(900, img.width)
                scale = display_width / img.width
                display_height = int(img.height * scale)
                img_resized = img.resize((display_width, display_height))

                existing_bounding_boxes = get_existing_value(
                    df_full, session_state.user, instance_id, 'bounding_box'
                )
                if existing_bounding_boxes:
                    boxes: list[Box] = [
                        json.loads(box) for box in existing_bounding_boxes
                    ]
                    img_resized = draw_boxes(img_resized, boxes)

                v = session_state.get('canvas_version', 0)

                canvas_result = st_canvas(
                    background_image=img_resized,  # pyright: ignore
                    drawing_mode='rect',
                    stroke_width=2,
                    stroke_color='#ff0000',
                    fill_color='rgba(255, 0, 0, 0.1)',
                    height=display_height,
                    width=display_width,
                    key=f'canvas-{instance_id}-{i}-{v}',
                )

                if canvas_result.json_data:
                    rects = [
                        o
                        for o in cast(
                            list[Rect], canvas_result.json_data['objects']
                        )
                        if o['type'] == 'rect'
                    ]
                    if rects:
                        boxes: list[Box] = [
                            {
                                'x': int(rect['left'] / scale),
                                'y': int(rect['top'] / scale),
                                'w': int(rect['width'] / scale),
                                'h': int(rect['height'] / scale),
                            }
                            for rect in rects
                        ]
                        if st.button(
                            'Save bounding box', key=f'save_bbox_{i}'
                        ):
                            for box in boxes:
                                append_annotation(
                                    AnnotationRow(
                                        name=session_state.user,
                                        instance_id=instance_id,
                                        issue_link=row['issue_link'],
                                        problem_statement=row[
                                            'problem_statement'
                                        ],
                                        image_assets=image_asset,
                                        key='bounding_box',
                                        value=json.dumps(box),
                                    )
                                )
                                st.success(f'Saved: {box}')

        st.divider()

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button('Log out', width='stretch'):
            del session_state.user
            st.rerun()
    prev_id = get_previous_incomplete(df_full, session_state.user, instance_id)
    with col2:
        if prev_id:
            if st.button('← Previous incomplete', width='stretch'):
                session_state.selected_instance = prev_id
                session_state.canvas_version = session_state.canvas_version + 1
                st.rerun()
    next_id = get_next_incomplete(df_full, session_state.user, instance_id)
    with col3:
        if next_id:
            if st.button('Next incomplete →', width='stretch'):
                session_state['selected_instance'] = next_id
                session_state['canvas_version'] = (
                    session_state.canvas_version + 1
                )
                st.rerun()

    if st.button('🔄 Refresh'):
        session_state['canvas_version'] = session_state.canvas_version + 1
        st.rerun()


st.set_page_config(layout='wide')
if 'user' not in session_state:
    login_screen()
else:
    home_screen()
