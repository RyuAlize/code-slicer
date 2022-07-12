

operator = ['{','}', '=','<<',':','?', '**','+','-','*','/','%','++','--','+=','|=','-=','*=','/=','==','!=','>','>=','<',"<=",'&&','||','!','&', '|', '~', '^','<<','>>','[',']','(',')', '.',';','->',',']

def checkHex(s: str) -> bool:  
    if s.startswith('0x') or s.startswith('0X'):
        s = s[2:] 
        for c in s:
            if not ( (c >= '0' and c <='9') or (c >='a' and c <= 'f') or (c >= 'A' and c <= 'F')):
                return False
        return True
    elif s.startswith('~0X') or s.startswith('~0x'):
        s = s[3:]
        for c in s:
            if not ( (c >= '0' and c <='9') or (c >='a' and c <= 'f') or (c >= 'A' and c <= 'F')):
                return False
        return True
    return False


def is_numic(token: str) -> bool :
    if token.isdigit():
        return True
    if checkHex(token):
        return True
    return False
    
def is_str_litrial(token: str) -> bool:
    if token.startswith('"') and token.endswith('"'):
        return True
    if token.startswith("'") and token.endswith("'"):
        return True
    return False

def is_identifier(token: str) -> bool:
    token = token.strip('"')
    for c in token:
        if not (c== '_' or (c >= '0' and c <='9') or (c >='a' and c <= 'z') or (c >= 'A' and c <= 'Z')):
            return False
    return True

def parse_token(token, txt_tokens: list):
    
    if token in operator:
        txt_tokens.append(token)
    elif is_numic(token):
        txt_tokens.append("<NUM>")
        #txt_tokens.append(token)
    elif is_str_litrial(token):
        txt_tokens.append("<STR>")
        #txt_tokens.append(token)
    elif is_identifier(token):
        token = token.strip('"')
        txt_tokens.append(token)
    else:
        token_list = []
        if token.endswith(';'):
            token_list = token.split(';')[:-1]
            token_list.append(';')

        elif len(token) > 1 and '(' in token:
            pos = token.index('(')
            token_list.append(token[:pos])
            token_list.append(token[pos])
            token_list.append(token[pos+1:])
        elif len(token) > 1 and ',' in token:
            pos = token.index(',')
            token_list.append(token[:pos])
            token_list.append(token[pos])
            token_list.append(token[pos+1:])
        elif len(token) > 1 and '.' in token:
            pos = token.index('.')
            token_list.append(token[:pos])
            token_list.append(token[pos])
            token_list.append(token[pos+1:])
        elif len(token) > 1 and ')' in token:
            pos = token.index(')')
            token_list.append(token[:pos])
            token_list.append(token[pos])
            token_list.append(token[pos+1:])
        elif len(token) > 1 and '[' in token:
            pos = token.index('[')
            token_list.append(token[:pos])
            token_list.append(token[pos])
            token_list.append(token[pos+1:])
        elif len(token) > 1 and '<' in token:
            pos = token.index('<')
            token_list.append(token[:pos])
            token_list.append(token[pos])
            token_list.append(token[pos+1:])
        elif '**' in token:
            pos = token.index('*')
            token_list.append(token[:pos])
            token_list.append(token[pos:pos+2])
            token_list.append(token[pos+2:])
        elif '*' in token and len(token) > 1:
            pos = token.index('*')
            token_list.append(token[:pos])
            token_list.append(token[pos])
            token_list.append(token[pos+1:])
        elif '&' in token and len(token) > 1:
            pos = token.index('&')
            token_list.append(token[:pos])
            token_list.append(token[pos])
            token_list.append(token[pos+1:])
        elif '->' in token and len(token) > 1:
            pos = token.index('->')
            token_list.append(token[:pos])
            token_list.append(token[pos:pos+2])
            token_list.append(token[pos+2:])
        
        for token in token_list:
            if token != '':
                parse_token(token, txt_tokens)