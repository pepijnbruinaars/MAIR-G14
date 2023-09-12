def match_sentence(sentence: str | int):
    restart_match = check_restart_matches(sentence)
    bye_match = check_bye_matches(sentence)
    if restart_match != 0:
        return restart_match
    
    if bye_match != 0:
        return bye_match
    
    return 0
    
def check_hello_matches(sentence: str):
    keywords = ['hi', 'hello']
    
    return keyword_check(sentence, 'hello', keywords)
    
def check_negate_matches(sentence: str):
    keywords = ['no']
    
    return keyword_check(sentence, 'negate', keywords)
    
def check_req_alts_matches(sentence: str):
    keywords = ['how about', 'what about', 'any other', 'else', 'another', 'is there', 'are there', 'other option']
    
    return keyword_check(sentence, 'reqalts', keywords)
    
def check_bye_matches(sentence: str):
    keywords = ['bye', 'goodbye', 'see you']
    
    return keyword_check(sentence, 'bye', keywords)
    
def check_restart_matches(sentence: str):
    keywords = ['restart', 'start over', 'start again', 'start from the beginning', 'start from scratch']
    
    return keyword_check(sentence, 'restart', keywords)
    

def keyword_check(sentence: str, label: str, keywords: list):
    # Check if any of the keywords are in a sentence
    for keyword in keywords:
        if keyword in sentence.lower():
            return label
    
    return 0

def main():
    print(match_sentence('bye'))
    
if __name__ == '__main__':
    main()