def match_sentence(sentence: str | int):
    # Find all matches
    reqmore_match = check_reqmore_matches(sentence)
    affirm_match = check_affirm_matches(sentence)
    ack_match = check_ack_matches(sentence)
    deny_match = check_deny_matches(sentence)
    thankyou_match = check_thankyou_matches(sentence)
    confirm_match = check_confirm_matches(sentence)
    inform_match = check_inform_matches(sentence)
    request_match = check_request_matches(sentence)
    repeat_match = check_repeat_matches(sentence)
    restart_match = check_restart_matches(sentence)
    bye_match = check_bye_matches(sentence)
    hello_match = check_hello_matches(sentence)
    negate_match = check_negate_matches(sentence)
    reqalts_match = check_req_alts_matches(sentence)
    
    if inform_match != 0:
        return inform_match
    
    if request_match != 0:
        return request_match
    
    if thankyou_match != 0:
        return thankyou_match
    
    if affirm_match != 0:
        return affirm_match
    
    if reqmore_match != 0:
        return reqmore_match
    
    if ack_match != 0:
        return ack_match
    
    if deny_match != 0:
        return deny_match
    
    if confirm_match != 0:
        return confirm_match
    
    if repeat_match != 0:
        return repeat_match
    
    if restart_match != 0:
        return restart_match
    
    if bye_match != 0:
        return bye_match
    
    if hello_match != 0:
        return hello_match
    
    if negate_match != 0:
        return negate_match
    
    if reqalts_match != 0:
        return reqalts_match
    
    # No matches found
    return 0

def check_request_matches(sentence: str):
    # Variations of words often used to request something
    keypairs = [['could', 'get'], ['can', 'get'], ['what', 'is'], ['what', 'does'],
                ['what', 'are'], ['what', 'do'], ['what', 'kind'], ['what', 'type'],
                ['may', 'i']]
    
    keypairs_check = keyword_check_combinations(sentence, 'request', keypairs)
    
    if keypairs_check != 0:
        return keypairs_check
    return 0

def check_thankyou_matches(sentence: str):
    # Variations of words often used to say thank you
    keywords = ['thanks', 'thank',]
    keypairs = [['thank', 'you']]
    
    keywords_check = keyword_check(sentence, 'thankyou', keywords)
    keypairs_check = keyword_check_combinations(sentence, 'thankyou', keypairs)
    
    if keypairs_check != 0:
        return keypairs_check
    if keywords_check != 0:
        return keywords_check
    return 0

def check_affirm_matches(sentence: str):
    # Variations of words often used to affirm a statement
    keywords = ['yes']
    keypairs = [['yeah', 'can'], ['yeah', 'looking'], ['yes', 'looking'], ['thats', 'it'], ['thats', 'ok']]
    
    keywords_check = keyword_check(sentence, 'affirm', keywords)
    keypairs_check = keyword_check_combinations(sentence, 'affirm', keypairs)
    
    if keypairs_check != 0:
        return keypairs_check
    if keywords_check != 0:
        return keywords_check
    return 0

def check_inform_matches(sentence: str):
    # Variations of words often used to inform a statement
    keywords = ['looking', 'need', 'serves']
    keypairs = [['cheap', 'food'], ['i', 'want'], ['can', 'have'], ['would', 'like'], ['can', 'find'], 
                ['are', 'there'], ['surprise', 'me'], ['serves', 'food'], ['doesnt', 'matter'],
                ['what', 'got']]
    
    keywords_check = keyword_check(sentence, 'inform', keywords)
    keypairs_check = keyword_check_combinations(sentence, 'inform', keypairs)
    
    if keypairs_check != 0:
        return keypairs_check
    if keywords_check != 0:
        return keywords_check
    return 0

def check_confirm_matches(sentence: str):
    # Variations of words often used to confirm a statement
    keypairs = [['is', 'it'], ['does', 'it'], ['is', 'that'], ['does', 'that'], ['is', 'this'], ['does', 'this']]
    
    return keyword_check_combinations(sentence, 'confirm', keypairs)

def check_reqmore_matches(sentence: str):
    keywords = ['more']
    
    return keyword_check(sentence, 'reqmore', keywords)

def check_deny_matches(sentence: str):
    # Variations of words often used to deny a statement
    keypairs = [['dont', 'want'], ['change'], ['no', 'not'], ['not', 'want'], 
                ['not', 'like'], ['not', 'need']]
    
    keypairs_check = keyword_check_combinations(sentence, 'deny', keypairs)
    
    if keypairs_check != 0:
        return keypairs_check
    return 0

def check_ack_matches(sentence: str):
    split_sentence = sentence.split()
    keywords = ['okay', 'kay']
    keypairs = [['got', 'it'], ['got', 'that'], ['thatll', 'do'], ['im', 'good']]
    
    keywords_check = keyword_check(sentence, 'ack', keywords)
    keypairs_check = keyword_check_combinations(sentence, 'ack', keypairs)
    
    if keypairs_check != 0:
        return keypairs_check
    if keywords_check != 0 and len(split_sentence) <= 2:
        return keywords_check
    return 0

def check_repeat_matches(sentence: str):
    # Variations of words often used to repeat a statement
    keywords = ['repeat']
    keypairs = [['say', 'again']]
    
    keywords_check = keyword_check(sentence, 'repeat', keywords)
    keypairs_check = keyword_check_combinations(sentence, 'repeat', keypairs)
    
    if keypairs_check != 0:
        return keypairs_check
    if keywords_check != 0:
        return keywords_check
    return 0

def check_hello_matches(sentence: str):
    # Variations of words often used to say hello
    keywords = ['hi', 'hello']
    keypairs = [['how', 'are'], ['how', 'doing']]
    
    keywords_check = keyword_check(sentence, 'hello', keywords)
    keypairs_check = keyword_check_combinations(sentence, 'hello', keypairs)
    
    if keypairs_check != 0:
        return keypairs_check
    if keywords_check != 0:
        return keywords_check
    return 0
    
def check_negate_matches(sentence: str):
    # Variations of words often used to negate a statement
    keywords = ['no']
    keypairs = [['not', 'this'], ['not', 'that'], ['not', 'these'], ['not', 'those'], ['not', 'sure']]
    
    keywords_check = keyword_check(sentence, 'negate', keywords)
    keypairs_check = keyword_check_combinations(sentence, 'negate', keypairs)
    
    if keypairs_check != 0:
        return keypairs_check
    if keywords_check != 0:
        return keywords_check
    return 0
    
def check_req_alts_matches(sentence: str):
    # Variations of words often used to ask for alternatives
    keywords = ['alternatives']
    keypairs = [['instead', 'this'], ['instead', 'that'], ['instead', 'these'], ['instead', 'those'],
                ['is', 'there', 'another'], ['are', 'there', 'other'], ['how', 'about'], ['what', 'about'],
                ['any', 'other'], ['other', 'option'], ['other', 'options']]
    
    keywords_check = keyword_check(sentence, 'reqalts', keywords)
    keypairs_check = keyword_check_combinations(sentence, 'reqalts', keypairs)
    
    if keypairs_check != 0:
        return keypairs_check
    if keywords_check != 0:
        return keywords_check
    return 0
    
def check_bye_matches(sentence: str):
    # Variations of words often used to say goodbye
    keywords = ['bye', 'goodbye']
    keypairs = [['see', 'you']]
    
    keywords_check = keyword_check(sentence, 'bye', keywords)
    keypairs_check = keyword_check_combinations(sentence, 'bye', keypairs)
    
    if keypairs_check != 0:
        return keypairs_check
    if keywords_check != 0:
        return keywords_check
    return 0
    
def check_restart_matches(sentence: str):
    # Variations of words often used to restart
    keywords = ['restart']
    keypairs = [['start', 'over'], ['start', 'again'], ['start', 'beginning'], ['start', 'scratch']]
    
    keywords_check = keyword_check(sentence, 'restart', keywords)
    keypairs_check = keyword_check_combinations(sentence, 'restart', keypairs)
    
    if keypairs_check != 0:
        return keypairs_check
    if keywords_check != 0:
        return keywords_check
    return 0
    
# Function which checks if any combination of the words in a list of lists is in a sentence
def keyword_check_combinations(sentence: str, label: str, keypairs: list):
    split_sentence = sentence.split()
    # Check if any of the keywords are in a sentence
    for keypair in keypairs:
        # Make sure all at least 2 of the words in the keypair are in the sentence
        if all(keyword in split_sentence for keyword in keypair):
            return label
    
    return 0

# Function which checks if any of the words in a list is in a sentence
def keyword_check(sentence: str, label: str, keywords: list):
    split_sentence = sentence.split()
    # Check if any of the keywords are in a sentence
    for keyword in keywords:
        # Make sure the keyword is not a substring of a word
        if keyword in split_sentence:
            return label
    
    return 0