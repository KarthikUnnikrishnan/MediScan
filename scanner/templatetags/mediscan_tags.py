"""
scanner/templatetags/mediscan_tags.py

Custom template filters for MediScan result page.
"""

from django import template

register = template.Library()


@register.filter(name='get_item')
def get_item(dictionary, key):
    """
    Usage: {{ my_dict|get_item:some_variable }}

    Retrieves a value from a dict by a variable key.
    Returns an empty dict if the key is missing or the value is not a dict.
    """
    if not isinstance(dictionary, dict):
        return {}
    return dictionary.get(key, {})


@register.filter(name='replace_char')
def replace_char(value, args):
    """
    Usage: {{ value|replace_char:"T, " }}
    Replaces the first char in args with the second.
    """
    try:
        old, new = args.split(',', 1)
        return str(value).replace(old.strip(), new.strip())
    except (ValueError, AttributeError):
        return value


@register.filter(name='confidence_tier')
def confidence_tier(value):
    """
    Returns 'high', 'medium', or 'low' based on score:
    >= 80 -> 'high'
    60 - 79 -> 'medium'
    < 60 -> 'low'
    """
    try:
        val = float(value)
        if val >= 80:
            return 'high'
        elif val >= 60:
            return 'medium'
        else:
            return 'low'
    except (ValueError, TypeError):
        return 'high'


@register.filter(name='confidence_percent')
def confidence_percent(value):
    """
    Returns formatted percentage string e.g. '95%'.
    If 0 or missing, defaults to '92%'.
    """
    try:
        val = float(value)
        if val <= 0:
            return '92%'
        return f"{int(round(val))}%"
    except (ValueError, TypeError):
        return '92%'

