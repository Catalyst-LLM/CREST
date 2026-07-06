import os
import sys
import argparse
import logging
import uuid
from typing import List, Dict
import yaml

from utils.llm_client import OpenAIClient, GeminiClient
from utils.file_utils import load_yaml, load_markdown, save_json, find_mineru_files, load_json
from utils.mineru import MinerUParser
from algorithms.data_structure import dicts_to_blocks
from algorithms.mllm_audit import audit_by_block_natural
from algorithms.document_extractor_planning import DocumentPlanningExtractor
from algorithms.document_planner import associate_blocks_llm
from algorithms.pipline_tools import identify_experiment_tables_llm, add_confidence_to_records_from_paper, add_evidence_to_records_from_paper

from collections import defaultdict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

global_data_dir = "cache/"

def main(llm_extractor: OpenAIClient,
         llm_judger: OpenAIClient,
         mllm_judger: OpenAIClient,
         extractor: DocumentPlanningExtractor,
         schema: Dict,
         mineru_json: str,
         pdf_path: str,
         use_table_classifier: bool,
         use_dynamic_context: bool = True,
         use_evidence: bool = False,
         use_confidence: bool = False,
         output_dir: str = "output"):
    # This function remains unchanged, exactly the same as original
    os.makedirs(output_dir, exist_ok=True)
    data_dir = os.path.join(output_dir, "data")
    image_cache_dir = os.path.join(output_dir, "images")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(image_cache_dir, exist_ok=True)

    parser = MinerUParser(pdf_path, mineru_json, cache_dir=image_cache_dir)
    blocks = parser.load_blocks()
    blocks = dicts_to_blocks(blocks)

    raw_records = []
    if use_table_classifier:
        if os.path.exists(os.path.join(data_dir, "exp_tables.json")):
            exp_idx = load_json(os.path.join(data_dir, "exp_tables.json"))
        else:
            exp_tables = identify_experiment_tables_llm(blocks, llm_extractor)
            print(f"Found {len(exp_tables)} experiment tables")
            exp_idx = [x.id for x in exp_tables]
            save_json(exp_idx, os.path.join(data_dir, "exp_tables.json"))
        exp_tables = [bk for i in exp_idx for bk in blocks if bk.id == i]

        raw_records = []
        for exp_table in exp_tables:
            if use_dynamic_context:
                if os.path.exists(os.path.join(data_dir, f"{exp_table.id}_linking.json")):
                    table_blocks_idx = load_json(os.path.join(data_dir, f"{exp_table.id}_linking.json"))
                else:
                    table_blocks_idx = associate_blocks_llm(exp_table, blocks, llm_extractor)
                    new_blocks = [block for block in blocks if block.id in table_blocks_idx]
                    save_json(table_blocks_idx, os.path.join(data_dir, f"{exp_table.id}_linking.json"))
                print(f"Found {len(table_blocks_idx)}: {table_blocks_idx} associated blocks for {exp_table.id}")

            if os.path.exists(os.path.join(data_dir, f"{exp_table.id}_records.json")):
                record = load_json(os.path.join(data_dir, f"{exp_table.id}_records.json"))
            else:
                paper_context = "\n".join([x.content for x in blocks])
                record = extractor.extract(paper_context, exp_table.content)
                raw_records.append(record)
                save_json(record, os.path.join(data_dir, f"{exp_table.id}_records.json"))

            if use_evidence:
                if os.path.exists(os.path.join(data_dir, f"{exp_table.id}_evidence.json")):
                    evidence = load_json(os.path.join(data_dir, f"{exp_table.id}_evidence.json"))
                else:
                    evidence = add_evidence_to_records_from_paper(record, blocks, schema, llm_judger)
                    save_json(evidence, os.path.join(data_dir, f"{exp_table.id}_evidence.json"))

                if os.path.exists(os.path.join(data_dir, f"{exp_table.id}_mllm_judger_result.json")):
                    mllm_judger_result = load_json(os.path.join(data_dir, f"{exp_table.id}_mllm_judger_result.json"))
                else:
                    mllm_judger_result = audit_by_block_natural(record, evidence, blocks, mllm_judger, schema, image_cache_dir)
                    save_json(mllm_judger_result, os.path.join(data_dir, f"{exp_table.id}_mllm_judger_result.json"))

            if use_confidence:
                if os.path.exists(os.path.join(data_dir, f"{exp_table.id}_confidence.json")):
                    confidence = load_json(os.path.join(data_dir, f"{exp_table.id}_confidence.json"))
                else:
                    confidence = add_confidence_to_records_from_paper(record, blocks, schema, llm_judger)
                    save_json(confidence, os.path.join(data_dir, f"{exp_table.id}_confidence.json"))

    save_json(raw_records, os.path.join(data_dir, "records.json"))


# ---------- New: Run from config file ----------
def run_from_config(config_path: str):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # Extract LLM configuration
    extractor_cfg = config['llm_extractor']
    judger_cfg = config['llm_judger']
    mllm_cfg = config['mllm_judger']

    llm_extractor = OpenAIClient(
        model_name=extractor_cfg['model_name'],
        base_url=extractor_cfg['base_url'],
        api_key=extractor_cfg['api_key']
    )
    llm_judger = OpenAIClient(
        model_name=judger_cfg['model_name'],
        base_url=judger_cfg['base_url'],
        api_key=judger_cfg['api_key']
    )
    mllm_judger = OpenAIClient(
        model_name=mllm_cfg['model_name'],
        base_url=mllm_cfg['base_url'],
        api_key=mllm_cfg['api_key']
    )

    # Load schema
    schema_path = config['schema_path']
    schema = load_json(schema_path)

    # Extract common parameters
    output_base = config.get('output_dir', 'output')
    use_table_classifier = config.get('use_table_classifier', True)
    use_dynamic_context = config.get('use_dynamic_context', True)
    use_evidence = config.get('use_evidence', False)
    use_confidence = config.get('use_confidence', False)
    use_cv = config.get('use_cv', False)

    input_cfg = config['input']
    mode = input_cfg.get('mode', 'batch')

    if mode == 'single':
        # Single document mode
        pdf_path = input_cfg['pdf_path']
        json_path = input_cfg['json_path']
        # Can specify output directory individually, otherwise use output_base
        output_dir = input_cfg.get('output_dir', output_base)
        extractor = DocumentPlanningExtractor(llm_extractor, schema, chunk_size=5)
        main(
            llm_extractor=llm_extractor,
            llm_judger=llm_judger,
            mllm_judger=mllm_judger,
            extractor=extractor,
            schema=schema,
            mineru_json=json_path,
            pdf_path=pdf_path,
            use_table_classifier=use_table_classifier,
            use_dynamic_context=use_dynamic_context,
            use_evidence=use_evidence,
            use_confidence=use_confidence,
            output_dir=output_dir
        )
    elif mode == 'batch':
        # Batch mode: iterate over each subfolder under mineru_result_dir
        mineru_result_dir = input_cfg['mineru_result_dir']
        # For batch mode, specify a base output directory; each document outputs to output_base/subfolder_name
        for folder_name in os.listdir(mineru_result_dir):
            sub_path = os.path.join(mineru_result_dir, folder_name)
            if not os.path.isdir(sub_path):
                continue
            pdf_path, json_path = find_mineru_files(sub_path)
            if not os.path.exists(json_path):
                logger.warning(f"Skipping {folder_name}, JSON file not found")
                continue
            output_dir = os.path.join(output_base, folder_name)
            os.makedirs(output_dir, exist_ok=True)
            extractor = DocumentPlanningExtractor(llm_extractor, schema, chunk_size=5)
            logger.info(f"Processing {folder_name}")
            main(
                llm_extractor=llm_extractor,
                llm_judger=llm_judger,
                mllm_judger=mllm_judger,
                extractor=extractor,
                schema=schema,
                mineru_json=json_path,
                pdf_path=pdf_path,
                use_table_classifier=use_table_classifier,
                use_dynamic_context=use_dynamic_context,
                use_evidence=use_evidence,
                use_confidence=use_confidence,
                output_dir=output_dir
            )
    else:
        raise ValueError(f"Unknown input.mode: {mode}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Document information extraction, run via configuration file")
    parser.add_argument("--config", "-c", required=True, help="Configuration file path (YAML format)")
    args = parser.parse_args()
    run_from_config(args.config)
