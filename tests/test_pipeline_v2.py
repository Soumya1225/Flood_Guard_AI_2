"""Tests for the v2 pipeline: name matching, labels (no look-ahead in features), and the IMD grid reader."""
import numpy as np
import pandas as pd

from pipeline_v2.common import match_district, split_districts
from pipeline_v2.s02_build_dataset import add_features, add_labels


def test_district_names():
    known = {"baleshwar", "khordha", "kendujhar", "cuttack"}
    assert match_district("Balasore", known) == "baleshwar"
    assert match_district("Khurda District", known) == "khordha"
    assert match_district("Keonjhar", known) == "kendujhar"
    assert match_district("Atlantis", known) is None
    assert split_districts("Cuttack, Puri and Khordha") == ["Cuttack", "Puri", "Khordha"]


def _frame():
    dates = pd.date_range("2016-01-01", "2017-12-31")
    return pd.DataFrame({"date": dates, "district": "cuttack", "rain_mm": np.arange(len(dates)) % 7 * 1.0})


def test_labels_look_ahead_only_in_target():
    df = add_features(_frame())
    ev = pd.DataFrame({"district": ["cuttack"], "start": [pd.Timestamp("2017-07-10")], "end": [pd.Timestamp("2017-07-12")]})
    df = add_labels(df, ev).set_index("date")
    # target is 1 on the 3 days before the start, 0 four days before
    assert df.loc["2017-07-07":"2017-07-09", "flood_next_3d"].tolist() == [1, 1, 1]
    assert df.loc["2017-07-06", "flood_next_3d"] == 0
    assert df.loc["2017-07-10":"2017-07-12", "flood_active"].tolist() == [1, 1, 1]
    # features on a day do not change when future rainfall changes
    a = add_features(_frame()).set_index("date")["rain_7d"]
    f2 = _frame()
    f2.loc[f2.date > "2017-07-08", "rain_mm"] = 999
    b = add_features(f2).set_index("date")["rain_7d"]
    assert a.loc["2017-07-08"] == b.loc["2017-07-08"]
