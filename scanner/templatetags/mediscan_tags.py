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
