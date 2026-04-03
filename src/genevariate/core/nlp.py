"""
NLP module for sample classification
"""

import ollama
from genevariate.config import CONFIG


def classify_sample(gsm_data: dict) -> dict:
    """Classifies sample - extracts condition, tissue, treatment, age, time, dosage.

    Args:
        gsm_data: dict or Series with sample metadata (must include GSM/gsm key)

    Returns:
        dict with GSM identifier and Classified_* fields
    """
    # Extract GSM identifier from input (handles both dict and DataFrame row)
    gsm_id = (gsm_data.get('GSM', None) or gsm_data.get('gsm', None)
              or gsm_data.get('geo_accession', 'Unknown'))
    if hasattr(gsm_id, 'item'):  # numpy scalar
        gsm_id = str(gsm_id)

    default_result = {
        'GSM': str(gsm_id),
        'Classified_Condition': 'unknown',
        'Classified_Tissue': 'unknown',
        'Classified_Treatment': 'unknown',
        'Classified_Age': None,
        'Classified_Time': None,
        'Classified_Dosage': None,
    }

    try:
        text_parts = []
        for key, value in gsm_data.items():
            if key.lower() in ['title', 'description', 'source', 'characteristics',
                               'gsm_title', 'source_name', 'characteristics_ch1',
                               'source_name_ch1']:
                val_str = str(value).strip()
                if val_str and val_str.lower() not in ('nan', 'none', ''):
                    text_parts.append(f"{key}: {val_str}")

        if not text_parts:
            return default_result

        full_text = "\n".join(text_parts)

        prompt = f"""Extract from this sample:

{full_text}

Return ONLY:
Condition: [disease or "healthy"]
Tissue: [organ/tissue]
Treatment: [drug or "none"]
Age: [e.g. "25 years" or blank]
Time: [e.g. "24h" or blank]
Dosage: [e.g. "10mg" or blank]"""

        model = CONFIG.get('ai', {}).get('model', 'gemma2:9b')

        # Safe Ollama call with response validation
        try:
            response = ollama.chat(
                model=model,
                messages=[{'role': 'user', 'content': prompt}],
                options={'temperature': 0.1, 'num_predict': 150},
            )
        except Exception as ollama_err:
            print(f"Ollama call failed for {gsm_id}: {ollama_err}")
            return default_result

        # Safely extract response text
        response_text = ""
        if hasattr(response, 'message') and hasattr(response.message, 'content'):
            response_text = (response.message.content or "").strip()
        elif isinstance(response, dict):
            response_text = response.get('message', {}).get('content', '').strip()

        if not response_text:
            return default_result

        result = dict(default_result)

        for line in response_text.split('\n'):
            line = line.strip()
            if line.startswith('Condition:'):
                val = line.replace('Condition:', '').strip()
                if val and val.lower() != 'unknown':
                    result['Classified_Condition'] = val
            elif line.startswith('Tissue:'):
                val = line.replace('Tissue:', '').strip()
                if val and val.lower() != 'unknown':
                    result['Classified_Tissue'] = val
            elif line.startswith('Treatment:'):
                val = line.replace('Treatment:', '').strip()
                if val and val.lower() not in ['unknown', 'none']:
                    result['Classified_Treatment'] = val
            elif line.startswith('Age:'):
                val = line.replace('Age:', '').strip()
                if val and val.lower() != 'unknown':
                    result['Classified_Age'] = val
            elif line.startswith('Time:'):
                val = line.replace('Time:', '').strip()
                if val and val.lower() != 'unknown':
                    result['Classified_Time'] = val
            elif line.startswith('Dosage:'):
                val = line.replace('Dosage:', '').strip()
                if val and val.lower() != 'unknown':
                    result['Classified_Dosage'] = val

        return result

    except Exception as e:
        print(f"Classification error for {gsm_id}: {e}")
        return default_result


def build_final_text(gsm_data: dict) -> str:
    """Build text from GSM metadata."""
    parts = []
    for key, value in gsm_data.items():
        if value:
            parts.append(f"{key}: {value}")
    return "\n".join(parts)


def get_comprehensive_gsm_text(gsm_data: dict) -> str:
    """Get comprehensive text from GSM data."""
    return build_final_text(gsm_data)
