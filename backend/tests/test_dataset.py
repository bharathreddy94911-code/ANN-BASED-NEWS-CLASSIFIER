import pytest
from app.ml.data_loader import dataset_manager
from app.config import LABEL_MAPPING, LABEL_NAMES

def test_dataset_loader_and_splits():
    ds = dataset_manager.load()
    assert 'train' in ds
    assert 'test' in ds
    assert len(ds['train']) == 120000
    assert len(ds['test']) == 7600
    
    # check columns
    assert 'text' in ds['train'].column_names
    assert 'label' in ds['train'].column_names

def test_label_mapping():
    info = dataset_manager.get_info()
    assert info['num_classes'] == 4
    assert info['label_mapping'] == LABEL_MAPPING
    assert info['classes'] == LABEL_NAMES

def test_dataset_pagination():
    res = dataset_manager.get_examples(split='train', page=1, page_size=5)
    assert res['total'] == 120000
    assert len(res['items']) == 5
    assert 'category' in res['items'][0]
