import json

def _parse_json_str(s: str) -> dict:
    """Robust JSON parser that falls back to yaml.safe_load for python dict strings and malformed JSON."""
    s = s.strip()
    if s.startswith('"') and s.endswith('"'):
        s = s[1:-1]
    elif s.startswith("'") and s.endswith("'"):
        s = s[1:-1]

    s = s.strip()
    # Unescape escaped quotes if bash stringified it
    if r'\"' in s:
        s = s.replace(r'\"', '"')

    try:
        return json.loads(s)
    except Exception:
        import yaml
        import re
        # Try to fix {key:value} to {key: value} for yaml, multiple times to catch all
        s_yaml = s
        prev = ""
        while s_yaml != prev:
            prev = s_yaml
            s_yaml = re.sub(r'([{,\s])([a-zA-Z0-9_]+):([^\s])', r'\1\2: \3', s_yaml)

        try:
            res = yaml.safe_load(s_yaml)
            if isinstance(res, dict):
                return res
            return {}
        except Exception:
            return {}
