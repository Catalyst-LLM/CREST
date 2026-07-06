import json
import re
from collections import defaultdict
from typing import List, Dict, Any, Optional
import numpy as np
from utils.llm_client import LLMClient, OpenAIClient   # Assuming your file is llm_client.py

# ==========================
# 1. Utility functions: unit extraction and appending
# ==========================
def extract_unit(value_str: str) -> str:
    match = re.search(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?\s*([a-zA-Z°%]+|g/mol|mol/L|kg/mol|MPa|kPa|bar|psi|min|h|s|ms|μmol|mmol|mol|L|mL|g|kg|mg|μg|°C|°F|K|%|ppm|ppb)', value_str)
    if match:
        return match.group(1).strip()
    return ""

def append_unit_to_definition(definition: str, value: str) -> str:
    unit = extract_unit(value)
    if unit and unit not in definition:
        if definition.endswith('.'):
            return f"{definition} Unit: {unit}."
        else:
            return f"{definition}. Unit: {unit}."
    return definition

# ==========================
# New: Statistics module
# ==========================
class SchemaStatistics:
    """
    Computes various statistical metrics from a list of normalized documents,
    for generating four figures in the paper.
    """

    def __init__(self, docs: List[Dict], pruning_threshold: float = 0.8, active_threshold: float = 0.2):
        """
        Args:
            docs: List of documents processed by ExtraFieldStandardizer, ordered by paper sequence.
            pruning_threshold: Fields with null rate above this threshold are considered 'prunable' (inactive).
            active_threshold: Fields with null rate below this threshold are considered 'active core fields' (high fill rate).
                               Note: active_threshold should be <= pruning_threshold, typically more strict.
        """
        self.docs = docs
        self.pruning_threshold = pruning_threshold
        self.active_threshold = active_threshold
        self._all_fields = None          # cache all field names
        self._field_null_rates = None    # cache final null rates

    def _extract_value(self, field_value: Any) -> Any:
        """Extract the actual data value from a field value (handles _is_extra dict)"""
        if isinstance(field_value, dict) and field_value.get('_is_extra'):
            return field_value.get('value')
        return field_value

    def _is_missing(self, value: Any) -> bool:
        """Check if a value is missing (None, empty string, empty list, etc.)"""
        if value is None:
            return True
        if isinstance(value, str) and value.strip() == "":
            return True
        if isinstance(value, (list, dict)) and len(value) == 0:
            return True
        if isinstance(value, float) and np.isnan(value):
            return True
        return False

    def _compute_field_null_rates(self, docs: List[Dict]) -> Dict[str, float]:
        """Compute the null rate for each field in the given document list"""
        field_counts = defaultdict(int)      # number of documents where the field appears
        field_missing = defaultdict(int)     # number of documents where the field is present but value is empty
        for doc in docs:
            for field, raw_val in doc.items():
                # Skip internal helper fields
                if field.startswith('_'):
                    continue
                field_counts[field] += 1
                val = self._extract_value(raw_val)
                if self._is_missing(val):
                    field_missing[field] += 1
            # For fields not present in this document, we count as missing later.
        # Fill missing counts: for any field, total docs minus presence count equals missing count.
        total_docs = len(docs)
        all_fields = set(field_counts.keys())
        null_rates = {}
        for field in all_fields:
            present = field_counts[field]
            missing_total = (total_docs - present) + field_missing.get(field, 0)
            null_rates[field] = missing_total / total_docs
        self._all_fields = all_fields
        return null_rates

    def get_null_rate_distribution(self) -> Dict[str, float]:
        """Return final null rates for all fields (for Figure 1)"""
        if self._field_null_rates is None:
            self._field_null_rates = self._compute_field_null_rates(self.docs)
        return self._field_null_rates

    def get_pruning_summary(self) -> Dict[str, Any]:
        """
        Return summary statistics related to pruning (corresponding to numeric labels in Figure 1)
        """
        null_rates = self.get_null_rate_distribution()
        total_fields = len(null_rates)
        if total_fields == 0:
            return {}
        candidates_pruning = sum(1 for r in null_rates.values() if r > self.pruning_threshold)
        candidates_retention = total_fields - candidates_pruning
        active_core = sum(1 for r in null_rates.values() if r <= self.active_threshold)
        return {
            "total_fields": total_fields,
            "candidates_for_pruning": candidates_pruning,
            "candidates_for_pruning_percent": candidates_pruning / total_fields * 100,
            "candidates_for_retention": candidates_retention,
            "candidates_for_retention_percent": candidates_retention / total_fields * 100,
            "active_core_fields": active_core,
            "active_core_fields_percent": active_core / total_fields * 100,
            "pruning_threshold": self.pruning_threshold,
            "active_threshold": self.active_threshold,
        }

    def compute_dynamics(self) -> Dict[str, List]:
        """
        Simulate the dynamic process in paper order, returning:
        - paper_indices: paper indices (1-based)
        - cumulative_discovered: cumulative number of distinct fields discovered
        - net_active: number of currently active fields (null rate <= pruning_threshold)
        - pruning_events: list of (paper_index, field_name)
        """
        cumulative_fields = set()
        active_fields = set()
        pruning_events = []
        cumulative_counts = []
        active_counts = []
        paper_indices = []

        pruned_recorded = set()   # to avoid duplicate recording of pruning for the same field

        for i, doc in enumerate(self.docs, start=1):
            current_fields = {k for k in doc.keys() if not k.startswith('_')}
            cumulative_fields.update(current_fields)

            cum_docs = self.docs[:i]
            null_rates = self._compute_field_null_rates(cum_docs)

            new_active = {f for f in cumulative_fields if null_rates.get(f, 1.0) <= self.pruning_threshold}
            for f in active_fields - new_active:
                if f not in pruned_recorded:
                    pruning_events.append((i, f))
                    pruned_recorded.add(f)
            active_fields = new_active

            cumulative_counts.append(len(cumulative_fields))
            active_counts.append(len(active_fields))
            paper_indices.append(i)

        return {
            "paper_indices": paper_indices,
            "cumulative_discovered": cumulative_counts,
            "net_active": active_counts,
            "pruning_events": pruning_events,
        }

    def compute_rank_displacement(self) -> List[Dict]:
        """
        Compute rank displacement for each field:
        - rank at first appearance (based on frequency among processed documents at that time)
        - final frequency rank (based on all documents)
        - displacement = final rank - first rank (positive means rank dropped, negative means rose)
        Returns a list of dicts with field, rank_at_discovery, rank_at_mature, displacement, first_appeared_at_paper.
        """
        total_docs = len(self.docs)
        # Final frequency: number of non-missing occurrences across all documents
        final_freq = {}
        for field in self.get_null_rate_distribution().keys():
            cnt = 0
            for doc in self.docs:
                if field in doc:
                    val = self._extract_value(doc[field])
                    if not self._is_missing(val):
                        cnt += 1
            final_freq[field] = cnt
        # Sort descending to get final ranks (1 = highest frequency)
        final_rank = {field: idx+1 for idx, (field, _) in enumerate(sorted(final_freq.items(), key=lambda x: x[1], reverse=True))}

        # Rank at first appearance
        first_rank = {}
        field_first_doc = {}
        for i, doc in enumerate(self.docs, start=1):
            current_fields = [f for f in doc.keys() if not f.startswith('_')]
            for f in current_fields:
                if f not in field_first_doc:
                    field_first_doc[f] = i
                    # Compute frequency of this field within the first i documents
                    freq_at_first = 0
                    for prev_doc in self.docs[:i]:
                        if f in prev_doc:
                            val = self._extract_value(prev_doc[f])
                            if not self._is_missing(val):
                                freq_at_first += 1
                    # Frequency distribution of all fields seen up to i
                    all_fields_up_to_i = set()
                    for prev_doc in self.docs[:i]:
                        all_fields_up_to_i.update([k for k in prev_doc.keys() if not k.startswith('_')])
                    freq_dict = {}
                    for ff in all_fields_up_to_i:
                        cnt = 0
                        for prev_doc in self.docs[:i]:
                            if ff in prev_doc:
                                val = self._extract_value(prev_doc[ff])
                                if not self._is_missing(val):
                                    cnt += 1
                        freq_dict[ff] = cnt
                    sorted_fields = sorted(freq_dict.items(), key=lambda x: x[1], reverse=True)
                    rank = {field: idx+1 for idx, (field, _) in enumerate(sorted_fields)}.get(f, len(sorted_fields)+1)
                    first_rank[f] = rank

        results = []
        for field in final_rank:
            if field in first_rank:
                disp = final_rank[field] - first_rank[field]
                results.append({
                    "field": field,
                    "rank_at_discovery": first_rank[field],
                    "rank_at_mature": final_rank[field],
                    "displacement": disp,
                    "first_appeared_at_paper": field_first_doc.get(field, 0)
                })
        return results

    def get_schema_state_at_n(self, n: int = 100) -> Dict[str, Any]:
        """
        Return the adaptive schema state after processing the first n papers:
        - schema_fields (active core): number of active core fields (null rate <= active_threshold)
        - non_schema_discovered_fields: number of discovered fields that are inactive (null rate > pruning_threshold)
        Note: 'non-schema discovered fields' are those that have been seen but never became core.
        """
        if n > len(self.docs):
            n = len(self.docs)
        docs_subset = self.docs[:n]
        null_rates = self._compute_field_null_rates(docs_subset)
        total_discovered = len(null_rates)
        active_core = sum(1 for r in null_rates.values() if r <= self.active_threshold)
        non_active = total_discovered - active_core
        return {
            "papers_processed": n,
            "total_discovered_fields": total_discovered,
            "active_core_fields": active_core,
            "active_core_percent": active_core / total_discovered * 100 if total_discovered else 0,
            "non_schema_discovered_fields": non_active,
            "non_schema_percent": non_active / total_discovered * 100 if total_discovered else 0,
        }

# ==========================
# 2. Cache manager (human-confirmed field mappings)
# ==========================
class FieldMappingCache:
    def __init__(self, cache_file_path: Optional[str] = None):
        self.mapping = {}          # {original_field_name: standard_field_name}
        self.cache_file_path = cache_file_path
        if cache_file_path:
            self.load(cache_file_path)

    def load(self, file_path: str):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                self.mapping = json.load(f)
            print(f"Successfully loaded cache mapping, total {len(self.mapping)} rules")
        except FileNotFoundError:
            print(f"Cache file {file_path} not found, will create new cache")
            self.mapping = {}
        except Exception as e:
            print(f"Failed to load cache: {e}")
            self.mapping = {}

    def save(self, file_path: Optional[str] = None):
        path = file_path or self.cache_file_path
        if not path:
            raise ValueError("Cache file path not specified")
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.mapping, f, indent=2, ensure_ascii=False)
        print(f"Cache saved to {path}")

    def get(self, original_field: str) -> Optional[str]:
        return self.mapping.get(original_field)

    def add(self, original_field: str, standard_field: str):
        self.mapping[original_field] = standard_field

    def record_unmapped(self, unmapped_fields: List[str], output_file: str = "unmapped_fields.json"):
        existing = set()
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                existing = set(json.load(f))
        except:
            pass
        new_fields = set(unmapped_fields) - existing
        if new_fields:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(list(existing | new_fields), f, indent=2, ensure_ascii=False)
            print(f"Recorded {len(new_fields)} unmapped fields to {output_file}")

# ==========================
# 3. Core agent class (cache-first + LLM fallback)
# ==========================
class ExtraFieldStandardizer:
    def __init__(self,
                 cache: FieldMappingCache,
                 llm_client: OpenAIClient = None,
                 standard_schema: Dict = None,
                 fallback_to_llm: bool = False,
                 batch_size: int = 10):
        """
        cache: FieldMappingCache instance
        llm_client: Your LLMClient instance (required if fallback_to_llm=True)
        standard_schema: Dictionary of standard field definitions, format {field_name: {"Description & StandardDefinition": "...", ...}}
        fallback_to_llm: Whether to call LLM for intelligent merging for fields not in cache
        batch_size: Batch size for LLM calls (only used when fallback_to_llm=True)
        """
        self.cache = cache
        self.llm_client = llm_client
        self.standard_schema = standard_schema or {}
        self.fallback_to_llm = fallback_to_llm
        self.batch_size = batch_size
        self.unmapped_fields_in_batch = set()

    def _is_extra_field(self, value: Any) -> bool:
        return isinstance(value, dict) and value.get("_is_extra") is True

    # ---------- Cache-based normalization ----------
    def _apply_cache_mapping(self, docs: List[Dict]) -> List[Dict]:
        for doc in docs:
            extra_items = [(k, v) for k, v in doc.items() if self._is_extra_field(v)]
            for orig_name, field_value in extra_items:
                target = self.cache.get(orig_name)
                if target:
                    del doc[orig_name]
                    if target in doc and self._is_extra_field(doc[target]):
                        existing = doc[target]
                        existing["evidence"] += f", {field_value.get('evidence', '')}"
                        existing.setdefault("_original_names", []).append(orig_name)
                    else:
                        new_field = field_value.copy()
                        new_field["_original_names"] = [orig_name]
                        if target in self.standard_schema:
                            std_def = self.standard_schema[target]["Description & StandardDefinition"]
                            value_str = new_field.get("value", "")
                            new_field["definition"] = append_unit_to_definition(std_def, value_str)
                        new_field["_is_extra"] = True
                        doc[target] = new_field
                else:
                    self.unmapped_fields_in_batch.add(orig_name)
                    field_value["_unmapped"] = True
        return docs

    # ---------- LLM fallback: intelligent merging for unmapped fields ----------
    def _call_llm_for_unmapped(self, docs: List[Dict]) -> Dict[str, str]:
        """
        Collect information of all unmapped fields and call LLM to get mappings.
        Returns {original_field_name: standard_field_name}
        """
        unmapped_info = []
        seen = set()
        for doc in docs:
            for k, v in doc.items():
                if self._is_extra_field(v) and v.get("_unmapped", False) and k not in seen:
                    seen.add(k)
                    unmapped_info.append({
                        "field_name": k,
                        "definition": v.get("definition", ""),
                        "example_value": v.get("value", "")
                    })
        if not unmapped_info:
            return {}

        # Build description of standard fields
        std_desc = "\n".join([
            f"- {name}: {info.get('Description & StandardDefinition', '')}"
            for name, info in self.standard_schema.items()
        ]) if self.standard_schema else "(No predefined standard fields)"

        extra_desc = "\n".join([
            f"- Field name: {info['field_name']}, Definition: {info['definition']}, Example value: {info['example_value']}"
            for info in unmapped_info
        ])

        prompt = f"""
You are a chemistry data standardization expert. Below is a set of standard fields (from a chemistry database schema) and a set of extra fields extracted from documents (marked with _is_extra).

Standard fields list (field name: definition):
{std_desc}

Extra fields list (to be mapped or kept):
{extra_desc}

Task:
1. For each extra field, determine whether it can be mapped to a standard field (semantically identical or very similar). If so, use the standard field name as "summery_name".
2. If it cannot be mapped to any standard field, you may create a new standard name (use lowercase with underscores, e.g., "catalyst_amount") and provide a reasonable definition.
3. If a standard field corresponds to multiple extra fields (synonyms), group them together, listing all original extra field names in "relation_name".
4. If an extra field is unique and cannot be grouped, also put it as a separate group; summery_name can reuse the original field name (but try to normalize it).

Output format: JSON array, each element:
{{"summery_name": "standard field name or new field name", "sumery_define": "definition of this attribute (use standard definition if mapped, otherwise provide based on context)", "relation_name": ["original extra field name1", "original extra field name2", ...]}}

Note:
- sumery_define should include necessary unit information (if the example value has a unit, indicate the unit in the definition).
- Output only the JSON array, no additional explanatory text.

Start your analysis.
"""
        response = self.llm_client.call(prompt)
        if not response:
            return {}
        parsed = self.llm_client.extract_json_from_response(response)
        if not isinstance(parsed, list):
            print(f"LLM did not return an array: {parsed}")
            return {}

        mapping = {}
        for group in parsed:
            target = group.get("summery_name")
            if not target:
                continue
            for orig in group.get("relation_name", []):
                mapping[orig] = target
        return mapping

    def _apply_llm_mapping(self, docs: List[Dict], llm_mapping: Dict[str, str]) -> List[Dict]:
        """Apply LLM mappings to documents and update cache"""
        for doc in docs:
            extra_items = [(k, v) for k, v in doc.items() if self._is_extra_field(v) and k in llm_mapping]
            for orig_name, field_value in extra_items:
                target = llm_mapping[orig_name]
                del doc[orig_name]
                if target in doc and self._is_extra_field(doc[target]):
                    existing = doc[target]
                    existing["evidence"] += f", {field_value.get('evidence', '')}"
                    existing.setdefault("_original_names", []).append(orig_name)
                else:
                    new_field = field_value.copy()
                    new_field["_original_names"] = [orig_name]
                    if target in self.standard_schema:
                        std_def = self.standard_schema[target]["Description & StandardDefinition"]
                        value_str = new_field.get("value", "")
                        new_field["definition"] = append_unit_to_definition(std_def, value_str)
                    new_field["_is_extra"] = True
                    doc[target] = new_field
                self.cache.add(orig_name, target)
        return docs

    # ---------- Statistics and batch processing ----------
    def _compute_batch_stats(self, docs: List[Dict]) -> Dict[str, Dict]:
        freq = defaultdict(int)
        def_map = {}
        type_map = {}
        for doc in docs:
            for key, value in doc.items():
                if self._is_extra_field(value):
                    freq[key] += 1
                    if key not in def_map:
                        def_map[key] = value.get("definition", "")
                        type_map[key] = value.get("suggested_type", "")
        summary = {}
        for field in freq:
            summary[field] = {
                "freq": freq[field],
                "definition": def_map.get(field, ""),
                "suggested_type": type_map.get(field, "")
            }
        return summary

    def _attach_summary(self, docs: List[Dict], summary: Dict[str, Dict]) -> List[Dict]:
        for doc in docs:
            doc["_extra_field_summary"] = summary
        return docs

    def process_batch(self, docs: List[Dict]) -> List[Dict]:
        """Process one batch: cache normalization -> optional LLM fallback -> statistics -> attach summary"""
        if not docs:
            return docs
        self.unmapped_fields_in_batch.clear()
        # 1. Cache normalization
        docs = self._apply_cache_mapping(docs)
        # 2. If LLM fallback is enabled and there are unmapped fields, call LLM and apply mapping
        if self.fallback_to_llm and self.unmapped_fields_in_batch and self.llm_client:
            llm_mapping = self._call_llm_for_unmapped(docs)
            if llm_mapping:
                docs = self._apply_llm_mapping(docs, llm_mapping)
                # Save updated cache
                self.cache.save()
        # 3. Record still-unmapped fields (maybe LLM didn't cover all)
        if self.unmapped_fields_in_batch:
            self.cache.record_unmapped(list(self.unmapped_fields_in_batch))
        # 4. Compute frequency summary
        summary = self._compute_batch_stats(docs)
        docs = self._attach_summary(docs, summary)
        return docs

    def process_all(self, all_docs: List[Dict]) -> List[Dict]:
        updated = []
        for i in range(0, len(all_docs), self.batch_size):
            batch = all_docs[i:i+self.batch_size]
            updated_batch = self.process_batch(batch)
            updated.extend(updated_batch)
        return updated

    @staticmethod
    def get_all_extra_fields(docs: List[Dict]) -> Dict[str, int]:
        freq = defaultdict(int)
        for doc in docs:
            for key, value in doc.items():
                if isinstance(value, dict) and value.get("_is_extra"):
                    freq[key] += 1
        return dict(freq)

# ==========================
# 4. Usage example
# ==========================
if __name__ == "__main__":
    # Assume you have an LLM client configuration
    llm_client = OpenAIClient(
        model_name="gpt-4o",
        base_url="https://api.openai.com/v1",  # Replace with your endpoint
        api_key="your-api-key"
    )

    # Load human-confirmed cache mapping
    cache = FieldMappingCache("field_cache.json")

    # Load standard schema (your provided JSON)
    with open("standard_schema.json", "r") as f:
        standard_schema = json.load(f)

    # Create standardizer agent with LLM fallback
    standardizer = ExtraFieldStandardizer(
        cache=cache,
        llm_client=llm_client,
        standard_schema=standard_schema,
        fallback_to_llm=True,
        batch_size=5
    )

    # Process a batch of documents
    sample_docs = [...]  # Your document list
    processed_docs = standardizer.process_all(sample_docs)

    # View results
    print(processed_docs)