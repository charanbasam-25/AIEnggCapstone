"""Shared styling for the Streamlit workspaces."""

import streamlit as st


def apply_style() -> None:
    st.markdown("""
    <style>
    .stApp { background: #f5f7fb; color: #17243b; }
    .block-container { max-width: 1160px; padding-top: 2.5rem; padding-bottom: 3rem; }
    [data-testid="stSidebar"] { background: #ffffff; border-right: 1px solid #e2e8f0; }
    h1, h2, h3 { letter-spacing: -0.035em; }
    [data-testid="stCaptionContainer"] { color: #64748b; }
    [data-testid="stVerticalBlockBorderWrapper"] > div { border-radius: 14px; }
    [data-testid="stMetric"] { background: #ffffff; border: 1px solid #e2e8f0;
        border-radius: 14px; padding: 18px 20px; }
    .stButton button { border-radius: 9px; min-height: 44px; font-weight: 600; }
    .stButton button[kind="primary"] { background: #2457d6; border-color: #2457d6; }
    [data-testid="stExpander"] { background: #ffffff; border-radius: 12px; }
    .studio-hero { background: #142744; border-radius: 20px; padding: 32px 36px;
        margin-bottom: 24px; color: #ffffff; box-sizing: border-box; }
    .studio-hero [data-testid="stHeadingWithActionElements"] {
        width: 100%; min-width: 0; }
    .studio-hero h1 { color: #ffffff; font-size: clamp(1.8rem, 4vw, 2.65rem);
        line-height: 1.15; margin: 8px 0 12px; padding: 0; width: auto;
        max-width: 100%; overflow-wrap: break-word; }
    .studio-hero p { color: #cbd8ec; max-width: min(740px, 100%);
        width: auto; margin: 0; line-height: 1.65; }
    .studio-eyebrow { font-size: 11px; font-weight: 700; letter-spacing: .16em;
        text-transform: uppercase; color: #9ebcf8; }
    .studio-option { background: white; border: 1px solid #e2e8f0; border-radius: 12px;
        padding: 16px; height: 100%; margin-bottom: 12px; line-height: 1.5; }
    .studio-option b { display: inline-block; color: #2457d6; margin-right: 10px; }
    .studio-brand { font-weight: 750; font-size: 24px; letter-spacing: -.04em; }
    .st-key-workspace_navigation [data-testid="stRadio"] label {
        width: 100%; padding: 10px 12px; border-radius: 10px;
        border: 1px solid transparent; margin-bottom: 4px; }
    .st-key-workspace_navigation [data-testid="stRadio"] label:has(input:checked) {
        background: #edf3ff; border-color: #dbe7ff; color: #2457d6; }
    .st-key-about_navigation { margin-bottom: 18px; }
    .st-key-about_navigation [data-testid="stRadio"] label {
        padding: 10px 16px; border: 1px solid #e2e8f0;
        border-radius: 10px; background: #ffffff; }
    .st-key-about_navigation [data-testid="stRadio"] label:has(input:checked) {
        background: #edf3ff; border-color: #93b0f3; color: #2457d6; }
    .about-steps { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 14px; margin: 20px 0 30px; }
    .about-step { padding: 20px; background: #ffffff; border: 1px solid #e2e8f0;
        border-radius: 14px; }
    .about-step-number { color: #2457d6; font-size: 12px; font-weight: 700; }
    .about-step h3 { font-size: 16px; letter-spacing: -.02em;
        margin: 12px 0 8px; padding: 0; }
    .about-step p { font-size: 13px; color: #64748b; line-height: 1.6; margin: 0; }
    [data-testid="stTable"] { max-width: 100%; overflow-x: auto; background: #ffffff;
        border-radius: 10px; }
    [data-testid="stTable"] table { width: 100%; min-width: 680px; table-layout: fixed; }
    [data-testid="stTable"] th, [data-testid="stTable"] td {
        white-space: normal; overflow-wrap: break-word; vertical-align: top;
        text-align: left; padding: 12px 14px; font-size: 14px; line-height: 1.6; }
    [data-testid="stTable"] th { background: #edf3ff; color: #243e63; }
    [data-testid="stTable"] th:first-child, [data-testid="stTable"] td:first-child {
        width: 22%; }
    .subject-icon { font-size: 26px; width: 46px; height: 46px;
        border-radius: 12px; background: #eef2f7; display: flex;
        align-items: center; justify-content: center; margin-bottom: 12px; }
    .subject-icon.available { background: #e8efff; color: #2457d6; }
    .subject-title { font-weight: 700; font-size: 17px; margin-bottom: 7px; }
    .subject-description { color: #64748b; font-size: 12px; min-height: 40px;
        line-height: 1.6; margin-bottom: 10px; }
    .subject-status { font-size: 11px; font-weight: 650; margin-bottom: 12px; }
    .subject-status.available { color: #2457d6; }
    .subject-status.locked { color: #64748b; }
    .st-key-subject_polity { background: #fff; border-color: #93b0f3; }
    [data-testid="stForm"] { background: #fff; border-radius: 16px; }
    [data-testid="stRadio"] label { line-height: 1.5; }
    [data-baseweb="tab-list"] { gap: 8px; }
    [data-baseweb="tab"] { padding: 12px 16px; }
    @media (max-width: 640px) {
        .block-container { padding-top: 1.25rem; }
        .studio-hero { padding: 24px; }
        .about-steps { grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px; }
        .about-step { padding: 16px; }
        .st-key-about_navigation [data-testid="stRadio"] label { padding: 8px 10px; }
        [data-baseweb="tab"] { padding: 10px; }
        .st-key-subject_catalog [data-testid="stHorizontalBlock"] {
            display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: .6rem; }
        .st-key-subject_catalog [data-testid="stColumn"] {
            width: auto; min-width: 0; }
        .st-key-subject_catalog [data-testid="stColumn"]:first-child {
            grid-column: 1 / -1; }
        .st-key-subject_catalog .subject-icon {
            float: left; font-size: 20px; width: 32px; height: 32px;
            margin: 0 10px 8px 0; }
        .st-key-subject_catalog .subject-title { font-size: 15px; padding-top: 4px; }
        .st-key-subject_catalog .subject-description,
        .st-key-subject_catalog .subject-status { display: none; }
    }
    </style>
    """, unsafe_allow_html=True)
