# Function to divide the review/helpfulness to give a loose estimate for reliability of review
import torch
def get_rating_reliability(value):
    numerator, denominator = value.split('/')
    if numerator == '0':
        return 0.0
    else:
        return float(numerator) / float(denominator)

def transform_to_nested_dict(df):
    # Grouping by 'Id' and 'Title', then applying the transformation
    grouped = df.groupby(['Id', 'Title'])
    result = grouped.apply(
        lambda x: {'review/summary': x['review/summary'].reset_index(drop=True).to_dict(), 
                    'review/text': x['review/text'].reset_index(drop=True).to_dict(),
                    'review/score': x['review/score'].reset_index(drop=True).to_dict()
                    })
    return result.reset_index(name='reviews')

def vectorize_texts(texts, tokenizer, model):
    # Tokenize the texts: convert them into a format the model can understand
    inputs = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt", add_special_tokens=True)

    # Perform the vectorization (forward pass through the model)
    with torch.no_grad():
        outputs = model(**inputs)

    # Extract the embeddings and return them
    return outputs.last_hidden_state.mean(dim=1).numpy()