import pytest
from app.ml.vocabulary import Vocabulary
from app.ml.cbow import CBOWModel
from app.ml.train import TrainingPipeline
from app.ml.evaluate import EvaluationPipeline
from app.ml.predict import Predictor

def test_smoke_end_to_end_training_and_inference():
    # Mini synthetic corpus
    train_texts = [
        "Oil prices surge as international stock market rallies strongly in Europe.",
        "Wall Street indexes close in the green with banking profits rising.",
        "Olympic soccer team secures victory in historic championship penalty shootout.",
        "Tennis star wins grand slam final with dominant serving performance.",
        "Tech giant launches breakthrough artificial intelligence computer processing chip.",
        "New software platform improves cybersecurity encryption for cloud servers."
    ]
    train_labels = [2, 2, 1, 1, 3, 3] # Business, Sports, Sci/Tech
    
    val_texts = [
        "Stock market shares advance amid global economic growth.",
        "Football championship concludes with exciting golden goal victory."
    ]
    val_labels = [2, 1]

    # 1. Preprocessing & Vocab
    vocab = Vocabulary(min_freq=1)
    vocab.build_from_tokens([t.lower().split() for t in train_texts])
    assert len(vocab) > 10

    # 2. Train CBOW embeddings
    embeds = TrainingPipeline.train_cbow_embeddings(
        train_texts=train_texts,
        vocab=vocab,
        embed_dim=16,
        window_size=1,
        epochs=1,
        batch_size=4,
        max_pairs=50
    )
    assert embeds.shape == (len(vocab), 16)

    # 3. Train Classifier
    model, history = TrainingPipeline.train_classifier(
        train_texts=train_texts,
        train_labels=train_labels,
        val_texts=val_texts,
        val_labels=val_labels,
        vocab=vocab,
        pretrained_embeddings=embeds,
        embed_dim=16,
        hidden_dim=16,
        epochs=1,
        batch_size=2,
        optimizer_name="sgd"
    )
    assert len(history['train_loss']) == 1

    # 4. Predict
    predictor = Predictor(model=model, vocab=vocab)
    pred_res = predictor.predict("Stock market tech investments rally")
    assert 'predicted_category' in pred_res
    assert 'confidence' in pred_res
    assert 'probabilities' in pred_res
    assert len(pred_res['probabilities']) == 4
