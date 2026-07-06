from collections import defaultdict
from typing import List, Dict, Any
import json
from .data_structure import Block
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def group_fields_by_block(
    records: List[Dict[str, Any]],
    evidence_list: List[Dict[str, str]]
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Group the fields of each record by the evidence block_id.

    Return format:
    {
        "28": [
            {"catalyst_ligand_type": "value1", "catalyst_name": "value2", ...},  # fields from record 0
            {"catalyst_ligand_type": "value3", "catalyst_name": "value4", ...},  # fields from record 1
            ...
        ],
        "14": [...]
    }
    Note: Each inner dict contains only the fields mapped to that block_id
          (may not include all fields of the original record).
    """
    block_to_records = defaultdict(list)

    for rec_idx, (record, evidence) in enumerate(zip(records, evidence_list)):
        # Collect fields of this record that are mapped to a specific block
        block_fields = defaultdict(dict)
        for field_name, value in record.items():
            if field_name.startswith("_"):
                continue
            block_id = evidence.get(field_name)
            if not block_id:
                continue
            block_fields[block_id][field_name] = value

        # Append each block subset of this record to the result
        for block_id, fields_dict in block_fields.items():
            # Optionally add _record_index for traceability
            fields_dict["_record_index"] = rec_idx
            block_to_records[block_id].append(fields_dict)

    return dict(block_to_records)


def convert_records_to_natural_language(records):
    display_records = []
    for rf in records:
        display_rf = {k: v for k, v in rf.items() if not k.startswith("_")}
        display_records.append(display_rf)

    records_json = json.dumps(display_records, indent=2, ensure_ascii=False)
    return records_json


def build_natural_prompt_for_block(
    block_text: str,
    records: List[Dict[str, Any]],
    audit_records: List[Dict[str, Any]],   # each element is a dict containing multiple fields and their values
    schema_str: Dict
) -> str:
    """
    Generate a natural-language-style auditing prompt.
    records_fields example:
    [
        {"catalyst_ligand_type": "X", "catalyst_name": "Y", "_record_index": 0},
        {"catalyst_ligand_type": "A", "catalyst_name": "B", "_record_index": 1}
    ]
    """
    # Remove internal _record_index before display
    records_json = convert_records_to_natural_language(records)
    audit_records_json = convert_records_to_natural_language(audit_records)

    prompt = f"""Based on the following image text:
{block_text}

The extracted experimental record part (JSON array, each element is a record):
{records_json}

The records to be audited:
{audit_records_json}

Field definitions (schema):
{schema_str}

Please review each field of each record to be audited based on the image.

Requirements:
- For each record, output only the fields that are **incorrect or need correction**. Do not output correct fields.
- Output format for each erroneous field: {{"field_name": ["suggested_correct_value", confidence]}}
  - Confidence ranges from 0.0 to 1.0, indicating your certainty that the original value is wrong and the suggested value is correct.
  - If the field is missing, malformed, or clearly inconsistent numerically, provide the most likely correct value; if uncertain, give null.
- Output is a JSON array with the same length as the input records. Each element is an object (may be empty).
- If all fields of a record are correct, output {{}}.

Example input records:
[
  {{"catalyst_name": "Pd/C", "temperature_c": "120"}},
  {{"catalyst_name": "Ni", "temperature_c": "80"}}
]

Example output:
[
  {{}},
  {{"catalyst_name": ["Pd(PPh3)4", 0.85], "temperature_c": ["100", 0.6]}}
]
(The second record has both fields wrong)

Output only JSON, no extra text.
"""
    return prompt


def audit_by_block_natural(
    records: List[Dict[str, Any]],
    evidence_list: List[Dict[str, str]],
    blocks: List[Block],
    mllm_client,
    schema_desc: Dict = {},
    image_cache_dir: str = None,
) -> List[Dict[str, Any]]:
    """
    Perform auditing by aggregating records per block, returning results in the original record order.
    """
    # 1. Group fields by block
    block_groups = group_fields_by_block(records, evidence_list)

    # 2. Build mapping block_id -> Block
    block_dict = {str(b.id): b for b in blocks}

    # 3. Store audit results, key = (record_index, field_name)
    audit_map = {}

    # 4. Call MLLM for each block
    for block_id, record_fields_list in block_groups.items():
        block = block_dict.get(str(block_id))

        if not block:
            logger.warning(f"Block {block_id} not found, skip {len(record_fields_list)} record groups")
            # Mark with low confidence
            for rf in record_fields_list:
                rec_idx = rf.get("_record_index")
                for fname, fval in rf.items():
                    if fname.startswith("_"):
                        continue
                    audit_map[(rec_idx, fname)] = {
                        "confidence": 0.0,
                        "suggested_value": None,
                        "reason": f"Block {block_id} not found",
                        "evidence_block_id": block_id
                    }
            continue

        # Prepare text and image context
        text_context = block.content if block.content else ""
        image = None

        has_image = image is not None
        # image_url = image_to_base64_data_uri(block.image_path) if has_image else None

        # Build prompt
        prompt = build_natural_prompt_for_block(text_context, records, record_fields_list, schema_desc)

        try:
            images = [image] if image else []
            response = mllm_client.call_with_image(prompt, block.image_path)

            parsed = mllm_client.extract_json_from_response(response)

            # Parse results
            for idx, record_result in enumerate(parsed):
                if idx >= len(record_fields_list):
                    print("idx >= len(record_fields_list)", idx, len(record_fields_list))
                    break
                rec_idx = record_fields_list[idx].get("_record_index")
                if rec_idx is None:
                    continue

                # Get the fields of this record that belong to the current block (fields to audit)
                fields_in_this_block = {k: v for k, v in record_fields_list[idx].items() if not k.startswith("_")}

                # Fields returned by the model as erroneous
                returned_fields = set(record_result.keys())

                # 1. Process explicit erroneous fields returned by the model
                for field_name, value in record_result.items():
                    if not isinstance(value, list) or len(value) < 2:
                        continue
                    suggested = value[0]
                    try:
                        confidence = float(value[1])
                    except:
                        confidence = 0.0
                    audit_map[(rec_idx, field_name)] = {
                        "confidence": confidence,
                        "suggested_value": suggested,
                        "reason": "Reviewed by MLLM (error)",
                        "evidence_block_id": block_id
                    }

                # 2. Process correct fields that the model did not flag
                for field_name in fields_in_this_block:
                    if field_name not in returned_fields:
                        audit_map[(rec_idx, field_name)] = {
                            "confidence": 1.0,
                            "suggested_value": None,
                            "reason": "Correct (not flagged by MLLM)",
                            "evidence_block_id": block_id
                        }
        except Exception as e:
            logger.error(f"Error auditing block {block_id}: {e}")
            for rf in record_fields_list:
                rec_idx = rf.get("_record_index")
                for fname, fval in rf.items():
                    if fname.startswith("_"):
                        continue
                    audit_map[(rec_idx, fname)] = {
                        "confidence": 0.0,
                        "suggested_value": None,
                        "reason": f"MLLM call failed: {e}",
                        "evidence_block_id": block_id
                    }

    # 5. Assemble final output in original record order
    final_results = []
    for rec_idx, record in enumerate(records):
        merged = {}
        # Preserve _record_id if present
        if "_record_id" in record:
            merged["_record_id"] = record["_record_id"]

        for field_name, original_value in record.items():
            if field_name.startswith("_"):
                continue
            key = (rec_idx, field_name)
            if key in audit_map:
                info = audit_map[key]
                merged[field_name] = {
                    "original_value": original_value,
                    "suggested_value": info["suggested_value"],
                    "confidence": info["confidence"],
                    "block_id": info["evidence_block_id"],
                    "reason": info["reason"]
                }
            else:
                merged[field_name] = {
                    "original_value": original_value,
                    "suggested_value": None,
                    "confidence": 0.0,
                    "block_id": None,
                    "reason": "No evidence or not reviewed"
                }
        final_results.append(merged)

    return final_results