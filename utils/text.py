import tiktoken

def get_tokenizer(model : str) :
    try :
        encoding = tiktoken.encoding_for_model(model)
        return encoding.encode
    except Exception as e:
        encoding = tiktoken.get_encoding("cl100k_base")
        return encoding.encode

def count_tokens(text : str , model : str) -> int:
    tokenizer = get_tokenizer(model)

    if tokenizer:
        return len(tokenizer(text))
    else:
        return estimate_tokens(text)

def estimate_tokens(text : str) -> int:
    return max(1 , len(text) // 4)
    