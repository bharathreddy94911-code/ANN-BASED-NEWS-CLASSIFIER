import pytest
import torch
from app.ml.preprocessing import clean_text, tokenize
from app.ml.vocabulary import Vocabulary
from app.ml.cbow import CBOWModel, generate_cbow_pairs
from app.ml.classifier import CBOWMLPClassifier

def test_preprocessing_and_tokenization():
    raw = "<b>Stock Market</b> rally on Wall St.\\b after tech surge!"
    cleaned = clean_text(raw)
    assert 'stock market' in cleaned
    assert '<b>' not in cleaned
    assert '\\' not in cleaned
    
    tokens = tokenize(raw)
    assert 'stock' in tokens
    assert 'market' in tokens

def test_vocabulary_encoding_decoding():
    corpus = [
        ['stock', 'market', 'rally'],
        ['oil', 'prices', 'surge', 'market']
    ]
    vocab = Vocabulary(min_freq=1)
    vocab.build_from_tokens(corpus)
    assert len(vocab) > 4
    
    encoded = vocab.encode(['stock', 'market', 'unknownword'], max_len=5)
    assert len(encoded) == 5
    assert encoded[0] == vocab.w2i['stock']
    assert encoded[2] == 1 # UNK token
    assert encoded[3] == 0 # PAD token
    assert encoded[4] == 0 # PAD token

def test_cbow_model_forward():
    vocab_size = 50
    embed_dim = 16
    cbow = CBOWModel(vocab_size=vocab_size, embed_dim=embed_dim)
    context_tokens = torch.tensor([[2, 3, 4, 5], [6, 7, 8, 9]], dtype=torch.long)
    logits = cbow(context_tokens)
    assert logits.shape == (2, vocab_size)

def test_classifier_forward_and_output_dim():
    vocab_size = 50
    embed_dim = 16
    for pool in ["mean", "max", "concat"]:
        model = CBOWMLPClassifier(vocab_size=vocab_size, embed_dim=embed_dim, hidden_dim=32, num_classes=4, pooling=pool)
        input_ids = torch.tensor([[2, 3, 0, 0], [4, 5, 6, 0]], dtype=torch.long)
        logits = model(input_ids)
        assert logits.shape == (2, 4), f"Failed for pooling mode: {pool}"

def test_inference_and_anti_collapse():
    from app.services.model_service import model_service
    
    # Check that model predicts diverse classes and does not collapse to single class
    test_cases = [
        ("The football club signed a talented striker ahead of the upcoming league season.", "Sports"),
        ("Wall Street stocks surged following quarterly corporate earnings reports.", "Business"),
        ("NASA launched a new robotic space probe to explore distant planetary systems.", "Sci/Tech"),
        ("The president signed international diplomacy peace treaties at the United Nations summit.", "World")
    ]
    
    predictions = []
    for text, expected in test_cases:
        res = model_service.predict(text)
        predictions.append(res["predicted_category"])
    
    # Verify diversity: at least 3 distinct categories predicted across diverse domains
    unique_preds = set(predictions)
    assert len(unique_preds) >= 3, f"Predictions collapsed: {predictions}"
    # Verify sports target is correctly classified as Sports
    sports_res = model_service.predict("The football club signed a talented striker ahead of the upcoming league season.")
    assert sports_res["predicted_category"] == "Sports"

