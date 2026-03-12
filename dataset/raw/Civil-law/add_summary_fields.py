import json
import os
import re

CIVIL_LAW_DIR = os.path.dirname(os.path.abspath(__file__))

def add_null_fields_after_summary(obj):
    """Recursively process a dict/list and insert the two null fields after 'summary'."""
    if isinstance(obj, list):
        return [add_null_fields_after_summary(item) for item in obj]
    if isinstance(obj, dict):
        new_obj = {}
        for key, value in obj.items():
            new_obj[key] = add_null_fields_after_summary(value)
            if key == "summary":
                # Insert the two new fields only if not already present
                if "legal_conditions_summary" not in obj:
                    new_obj["legal_conditions_summary"] = None
                if "penalties_summary" not in obj:
                    new_obj["penalties_summary"] = None
        return new_obj
    return obj

json_files = [f for f in os.listdir(CIVIL_LAW_DIR) if f.endswith(".json")]

for filename in json_files:
    filepath = os.path.join(CIVIL_LAW_DIR, filename)
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        updated = add_null_fields_after_summary(data)

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(updated, f, ensure_ascii=False, indent=4)

        print(f"✓ Updated: {filename}")
    except Exception as e:
        print(f"✗ Error processing {filename}: {e}")

print("\nDone!")
