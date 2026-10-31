import pytest

from imc.data import list_images


def test_missing_data_reports_root_and_remedy(tmp_path):
    with pytest.raises(FileNotFoundError, match='Pass --root') as error:
        list_images(root=tmp_path)
    assert str((tmp_path / 'train').resolve()) in str(error.value)
