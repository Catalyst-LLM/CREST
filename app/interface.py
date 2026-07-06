# streamlit_app_complete.py
import os
import json
import streamlit as st
import pandas as pd
import plotly.express as px
import sys
import re
from typing import List, Dict, Any, Optional

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import core modules (ensure these exist in your project)
from utils.llm_client import OpenAIClient, GeminiClient
from utils.file_utils import load_json, save_json
from utils.mineru import MinerUParser


# ---------- Helper Functions ----------
def load_json_file(path: str) -> Any:
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None

def load_table_data(data_dir: str, table_id: str):
    records = load_json_file(os.path.join(data_dir, f"{table_id}_records.json")) or []
    evidence = load_json_file(os.path.join(data_dir, f"{table_id}_evidence.json")) or []
    confidence = load_json_file(os.path.join(data_dir, f"{table_id}_confidence.json")) or []
    mllm_audit = load_json_file(os.path.join(data_dir, f"{table_id}_mllm_judger_result.json")) or []
    linking = load_json_file(os.path.join(data_dir, f"{table_id}_linking.json")) or []
    return records, evidence, confidence, mllm_audit, linking

def build_block_image_map_from_dir(image_dir: str) -> Dict[str, str]:
    """
    Read all image files from the images directory, extract block ID according to naming convention.
    Rule: filename contains _b{number}_, e.g. image_p0_b1_212_695_1675_808.png -> block ID is "1"
    Returns {block_id: image_path} mapping.
    """
    block_map = {}
    if not os.path.isdir(image_dir):
        return block_map
    pattern = re.compile(r'_b(\d+)_')  # match _bnumber_
    for fname in os.listdir(image_dir):
        if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            match = pattern.search(fname)
            if match:
                block_id = match.group(1)  # numeric string
                block_map[block_id] = os.path.join(image_dir, fname)
    return block_map

def flatten_record(record: Dict) -> Dict:
    flat = {}
    for k, v in record.items():
        if isinstance(v, dict) and v.get("_is_extra"):
            flat[k] = v.get("value")
        else:
            flat[k] = v
    return flat

def get_record_images(record_idx: int, evidence: List[Dict], mllm_audit: List[Dict], block_image_map: Dict) -> List[str]:
    """
    Extract block IDs associated with all fields of this record from evidence or mllm_audit,
    return corresponding image paths.
    Priority: block_id from mllm_audit (if exists) > evidence value.
    """
    block_ids = set()
    if record_idx < len(mllm_audit):
        audit = mllm_audit[record_idx]
        for field_info in audit.values():
            if isinstance(field_info, dict) and "block_id" in field_info:
                bid = str(field_info["block_id"])
                if bid:
                    block_ids.add(bid)
    if not block_ids and record_idx < len(evidence):
        ev = evidence[record_idx]
        for field, bid in ev.items():
            if isinstance(bid, str) and bid:
                block_ids.add(bid)
    images = []
    for bid in block_ids:
        img_path = block_image_map.get(bid)
        if img_path and os.path.exists(img_path):
            images.append(img_path)
        else:
            st.session_state.setdefault("missing_block_ids", set()).add(bid)
    return images

def build_confidence_heatmap(confidence_list: List[Dict]) -> Optional[px.imshow]:
    if not confidence_list:
        return None
    df = pd.DataFrame(confidence_list)
    numeric_cols = df.select_dtypes(include=['number']).columns
    if numeric_cols.empty:
        return None
    fig = px.imshow(df[numeric_cols].T,
                    text_auto=True,
                    title="Field Confidence Heatmap",
                    color_continuous_scale="RdYlGn",
                    aspect="auto")
    return fig

def load_document_results(doc_dir: str) -> dict:
    """
    Load all table data for a single document directory. Returns a dict containing
    table_ids, tables_data, block_image_map. Returns None if directory is invalid.
    """
    doc_dir = os.path.abspath(doc_dir)
    data_dir = os.path.join(doc_dir, "data")
    exp_tables_path = os.path.join(data_dir, "exp_tables.json")
    if not os.path.isdir(doc_dir) or not os.path.exists(exp_tables_path):
        return None

    table_ids = load_json_file(exp_tables_path) or []
    image_dir = os.path.join(doc_dir, "images")
    block_image_map = build_block_image_map_from_dir(image_dir)
    tables_data = {}
    for tid in table_ids:
        tid_str = str(tid)
        records, evidence, confidence, mllm_audit, linking = load_table_data(data_dir, tid_str)
        tables_data[tid_str] = {
            "records": records,
            "evidence": evidence,
            "confidence": confidence,
            "mllm_audit": mllm_audit,
            "linking": linking
        }
    return {
        "table_ids": [str(tid) for tid in table_ids],
        "tables_data": tables_data,
        "block_image_map": block_image_map,
        "image_dir": image_dir
    }

def scan_documents(root_dir: str) -> List[str]:
    """Scan root directory, return absolute paths of all subdirectories that contain data/exp_tables.json, sorted by name."""
    root_dir = os.path.abspath(root_dir)
    if not os.path.isdir(root_dir):
        return []
    doc_dirs = []
    for item in os.listdir(root_dir):
        sub_path = os.path.join(root_dir, item)
        if os.path.isdir(sub_path) and os.path.exists(os.path.join(sub_path, "data", "exp_tables.json")):
            doc_dirs.append(sub_path)
    return sorted(doc_dirs)

# Load saved user choices
def load_user_choices(doc_dir: str) -> Dict:
    choices_path = os.path.join(doc_dir, "user_choices.json")
    if os.path.exists(choices_path):
        return load_json_file(choices_path) or {}
    return {}

# ---------- Streamlit UI ----------
st.set_page_config(page_title="Chemical Experiment Extraction System", layout="wide")
st.title("🧪 Catalyst Experiment Multimodal Review System")

# Initialize session state
if "doc_dirs" not in st.session_state:
    st.session_state.doc_dirs = []
if "current_doc_idx" not in st.session_state:
    st.session_state.current_doc_idx = 0
if "doc_loaded" not in st.session_state:
    st.session_state.doc_loaded = False
if "table_ids" not in st.session_state:
    st.session_state.table_ids = []
if "tables_data" not in st.session_state:
    st.session_state.tables_data = {}
if "block_image_map" not in st.session_state:
    st.session_state.block_image_map = {}
if "user_choices" not in st.session_state:
    st.session_state.user_choices = {}
if "current_doc_dir" not in st.session_state:
    st.session_state.current_doc_dir = ""
if "missing_block_ids" not in st.session_state:
    st.session_state.missing_block_ids = set()
if "show_debug" not in st.session_state:
    st.session_state.show_debug = False

# Sidebar configuration
with st.sidebar:
    st.header("Document Navigation")
    root_dir = st.text_input("Root directory containing document folders", value="outputs")
    
    if st.button("Scan documents", type="primary"):
        dirs = scan_documents(root_dir)
        if not dirs:
            st.error(f"No valid document directories found under {root_dir}")
        else:
            st.session_state.doc_dirs = dirs
            st.session_state.current_doc_idx = 0
            st.session_state.doc_loaded = False
            st.success(f"Found {len(dirs)} documents")
            st.rerun()
    
    st.session_state.show_debug = st.checkbox("Show debug info", value=st.session_state.show_debug)
    
    if st.session_state.doc_dirs:
        total_docs = len(st.session_state.doc_dirs)
        current_idx = st.session_state.current_doc_idx
        current_doc_name = os.path.basename(st.session_state.doc_dirs[current_idx]) if total_docs > 0 else ""
        st.markdown(f"**Document {current_idx+1} of {total_docs}**")
        st.markdown(f"📄 **{current_doc_name}**")
        
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            if st.button("◀ Previous", disabled=(current_idx == 0)):
                st.session_state.current_doc_idx = current_idx - 1
                st.session_state.doc_loaded = False
                st.rerun()
        with col2:
            new_idx = st.number_input("Go to", min_value=1, max_value=total_docs, value=current_idx+1, step=1, label_visibility="collapsed")
            if new_idx != current_idx+1:
                st.session_state.current_doc_idx = new_idx - 1
                st.session_state.doc_loaded = False
                st.rerun()
        with col3:
            if st.button("Next ▶", disabled=(current_idx == total_docs-1)):
                st.session_state.current_doc_idx = current_idx + 1
                st.session_state.doc_loaded = False
                st.rerun()
        
        if not st.session_state.doc_loaded:
            doc_path = st.session_state.doc_dirs[st.session_state.current_doc_idx]
            with st.spinner(f"Loading document: {os.path.basename(doc_path)}..."):
                result = load_document_results(doc_path)
                if result:
                    st.session_state.table_ids = result["table_ids"]
                    st.session_state.tables_data = result["tables_data"]
                    st.session_state.block_image_map = result["block_image_map"]
                    st.session_state.current_doc_dir = doc_path
                    st.session_state.doc_loaded = True
                    st.session_state.missing_block_ids = set()
                    # Load saved user choices
                    saved_choices = load_user_choices(doc_path)
                    st.session_state.user_choices = saved_choices
                    st.success("Loaded successfully")
                    st.rerun()
                else:
                    st.error("Failed to load document data")
                    st.session_state.doc_loaded = False
    else:
        st.info("Please scan a root directory to find documents.")

# Main area: display current document results
if st.session_state.doc_loaded and st.session_state.table_ids:
    table_ids = st.session_state.table_ids
    tables_data = st.session_state.tables_data
    block_image_map = st.session_state.block_image_map
    current_doc_name = os.path.basename(st.session_state.current_doc_dir)
    st.subheader(f"📄 Current Document: {current_doc_name}")
    
    if st.session_state.show_debug:
        with st.expander("Debug Info", expanded=False):
            st.write(f"**Image directory:** {os.path.join(st.session_state.current_doc_dir, 'images')}")
            st.write(f"**Total block images found:** {len(block_image_map)}")
            if block_image_map:
                st.write("**Sample block IDs -> paths:**", dict(list(block_image_map.items())[:5]))
            st.write(f"**Missing block IDs (if any):** {st.session_state.missing_block_ids}")
    
    if not table_ids:
        st.info("This document has no experiment tables.")
    else:
        tabs = st.tabs([f"Table {tid}" for tid in table_ids])
        for tab_idx, tid in enumerate(table_ids):
            with tabs[tab_idx]:
                data = tables_data[tid]
                records = data["records"]
                mllm_audit = data["mllm_audit"]
                evidence = data["evidence"]
                confidence = data["confidence"]
                linking = data["linking"]

                if linking:
                    st.subheader(f"Table {tid} associated images (blocks: {linking})")
                    img_list = []
                    cap_list = []
                    missing_blocks = []
                    for bid in linking:
                        bid_str = str(bid)
                        img_path = block_image_map.get(bid_str)
                        if img_path and os.path.exists(img_path):
                            img_list.append(img_path)
                            cap_list.append(f"Block {bid}")
                        else:
                            missing_blocks.append(bid_str)
                    if img_list:
                        st.image(img_list, width=400, caption=cap_list)
                    else:
                        st.info("No images found for the associated blocks.")
                    if missing_blocks and st.session_state.show_debug:
                        st.warning(f"Missing images for blocks: {missing_blocks}")
                else:
                    st.subheader(f"Table {tid} records")

                if records:
                    flat_records = [flatten_record(r) for r in records]
                    df = pd.DataFrame(flat_records)
                    st.dataframe(df, use_container_width=True)

                    for rec_idx, rec in enumerate(records):
                        with st.expander(f"Record {rec_idx+1}: {rec.get('catalyst_name', 'Unknown')}"):
                            # Layout: left column for text, right column for images
                            col_left, col_right = st.columns([2, 1])
                            
                            with col_left:
                                # Get confidence dict for this record
                                conf_dict = confidence[rec_idx] if rec_idx < len(confidence) else {}
                                
                                st.markdown("**Field Value Selection**")
                                for field, llm_value in rec.items():
                                    if field.startswith("_"):
                                        continue
                                    mllm_suggested = None
                                    if rec_idx < len(mllm_audit):
                                        field_audit = mllm_audit[rec_idx].get(field, {})
                                        if isinstance(field_audit, dict):
                                            mllm_suggested = field_audit.get("suggested_value")
                                    
                                    choice_key = f"{tid}_{rec_idx}_{field}"
                                    current_choice = st.session_state.user_choices.get(choice_key, "LLM")
                                    
                                    # Get field confidence if exists
                                    field_conf = conf_dict.get(field) if isinstance(conf_dict, dict) else None
                                    conf_str = f" (conf: {field_conf:.2f})" if isinstance(field_conf, (int, float)) else ""
                                    
                                    # Check if values differ
                                    different = (mllm_suggested is not None and str(mllm_suggested) != str(llm_value))
                                    
                                    if different:
                                        # Show two options using nested columns
                                        st.markdown(f"**{field}**{conf_str}")
                                        opt_col1, opt_col2 = st.columns(2)
                                        with opt_col1:
                                            # LLM option, light blue background
                                            st.markdown(
                                                f"<div style='background-color: #d4eaf7; padding: 5px; border-radius: 5px;'>"
                                                f"📝 LLM: `{llm_value}`</div>",
                                                unsafe_allow_html=True
                                            )
                                        with opt_col2:
                                            # MLLM option, light orange background
                                            st.markdown(
                                                f"<div style='background-color: #ffe6cc; padding: 5px; border-radius: 5px;'>"
                                                f"🤖 MLLM: `{mllm_suggested}`</div>",
                                                unsafe_allow_html=True
                                            )
                                        # Radio button to choose
                                        choice = st.radio(
                                            "Choose which value to use",
                                            options=["LLM", "MLLM"],
                                            index=0 if current_choice == "LLM" else 1,
                                            key=f"radio_{choice_key}",
                                            horizontal=True,
                                            label_visibility="collapsed"
                                        )
                                        st.session_state.user_choices[choice_key] = choice
                                    else:
                                        # Values agree, show single value
                                        if mllm_suggested is not None:
                                            st.write(f"**{field}**{conf_str}: `{llm_value}` (LLM & MLLM agree)")
                                        else:
                                            st.write(f"**{field}**{conf_str}: `{llm_value}` (LLM only)")
                            
                            with col_right:
                                # Display associated images
                                rec_images = get_record_images(rec_idx, evidence, mllm_audit, block_image_map)
                                if rec_images:
                                    for img in rec_images:
                                        st.image(img, width=300, caption="Associated image")
                                else:
                                    st.caption("No associated images")
                                    if st.session_state.show_debug:
                                        st.write("Debug: evidence for this record:", evidence[rec_idx] if rec_idx < len(evidence) else "None")
                else:
                    st.info(f"Table {tid} has no records.")

                if confidence:
                    st.subheader("Confidence Heatmap")
                    fig = build_confidence_heatmap(confidence)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("Confidence data format is incorrect.")
                else:
                    st.info("No confidence data found.")

    # Export functionality
    st.sidebar.markdown("---")
    st.sidebar.subheader("Export for current document")
    
    # Reload saved choices button
    if st.sidebar.button("Reload saved choices"):
        saved = load_user_choices(st.session_state.current_doc_dir)
        st.session_state.user_choices = saved
        st.sidebar.success(f"Loaded {len(saved)} choices")
        st.rerun()
    
    if st.sidebar.button("Generate final records"):
        final_records_by_table = {}
        for tid, data in tables_data.items():
            records = data["records"]
            mllm_audit = data["mllm_audit"]
            final_records = []
            for rec_idx, rec in enumerate(records):
                final_rec = rec.copy()
                for field in rec.keys():
                    if field.startswith("_"):
                        continue
                    choice_key = f"{tid}_{rec_idx}_{field}"
                    choice = st.session_state.user_choices.get(choice_key, "LLM")
                    if choice == "MLLM" and rec_idx < len(mllm_audit):
                        suggested = mllm_audit[rec_idx].get(field, {}).get("suggested_value")
                        if suggested is not None:
                            final_rec[field] = suggested
                final_records.append(final_rec)
            final_records_by_table[tid] = final_records
        out_path = os.path.join(st.session_state.current_doc_dir, "final_records_by_table.json")
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(final_records_by_table, f, indent=2, ensure_ascii=False)
        st.sidebar.success(f"Saved to {out_path}")
        st.sidebar.download_button(
            label="Download final records",
            data=json.dumps(final_records_by_table, indent=2, ensure_ascii=False),
            file_name="final_records_by_table.json",
            mime="application/json"
        )

    if st.sidebar.button("Save user choices"):
        choices_path = os.path.join(st.session_state.current_doc_dir, "user_choices.json")
        with open(choices_path, 'w', encoding='utf-8') as f:
            json.dump(st.session_state.user_choices, f, indent=2, ensure_ascii=False)
        st.sidebar.success(f"Saved to {choices_path}")

    st.sidebar.download_button(
        label="Download user choices",
        data=json.dumps(st.session_state.user_choices, indent=2, ensure_ascii=False),
        file_name="user_choices.json",
        mime="application/json"
    )
else:
    if not st.session_state.doc_dirs:
        st.info("👈 Please scan a root directory containing document folders (each with data/exp_tables.json).")
    else:
        st.warning("No document loaded. Try scanning again or selecting a valid document.")