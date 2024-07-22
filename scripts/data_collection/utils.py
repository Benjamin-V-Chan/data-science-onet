import sys

def get_user_input(prompt):
    result = ''
    while len(result) == 0:
        result = input(prompt + ': ').strip()
    return result

def check_for_error(service_result):
    if 'error' in service_result:
        sys.exit(service_result['error'])