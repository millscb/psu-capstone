import os
import warnings

import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from htmrl.encoder_layer.rdse import RandomDistributedScalarEncoder
from htmrl.input_layer.improved_input_handler import InputHandler

warnings.simplefilter(action="ignore", category=FutureWarning)

"""
    Why do I have so much here? A lot of this can be turned into integration tests and or unit tests.
    I am keeping this for future test writing.
"""


def main():
    handler = InputHandler.get_instance()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(project_root, "data", "rec-center-hourly.csv")
    handler.input_data(input_source=data_path, required_columns=[])
    input_records = handler.to_records()

    encoder = RandomDistributedScalarEncoder(RDSEParameters())

    def _numeric_column(records: list[dict], col: str) -> list[float]:
        out = []
        for r in records:
            v = r.get(col)
            if v is None:
                continue
            try:
                out.append(float(v))
            except Exception:
                continue
        return out

    values = _numeric_column(input_records, "kw_energy_consumption")
    train_values, test_values = train_test_split(values, test_size=0.2, shuffle=False)
    # Train the knn for decoding on 80% of data
    encodings = []
    for v in train_values:
        encodings.append(encoder.encode(v))

    # Get SDRs from a different RDSE for 20% test data
    # I use a second RDSE so the first one that is already
    # trained does not train on the test sdrs.
    test_encodings = []
    encoder2 = RandomDistributedScalarEncoder(RDSEParameters())
    for v in test_values:
        test_encodings.append(encoder2.encode(v))

    # Decode the test SDRs and compare to the test values
    test_decoding = []
    for v in test_encodings:
        test_decoding.append(encoder.decode_knn(v))
    # for v, d in zip(test_values, test_decoding):
    #    print("Value encoded: ", v)
    #    print("Value decoded: ", d)
    rmse = np.sqrt(mean_squared_error(test_values, test_decoding))
    mi = min(values)
    ma = max(values)
    print("Column max: ", ma)
    print("Column min: ", mi)
    print("RMSE:", rmse)
    # =======================================================================================================

    # values = [float(row["Wave"]) for row in input_records if row.get("Wave") is not None]
    # data_path = os.path.join(project_root, "data", "test.csv")
    # =======================================================================================================
    """
    encoder_htm = RandomDistributedScalarEncoder()
    # HTM testing here
    # train_values, test_values = train_test_split(values, test_size=0.20, shuffle=False)
    train_values = values
    sdrs_train = []
    # Train sdrs for HTM
    for v in train_values:
        sdrs_train.append(encoder_htm.encode(v))
    # Test sdrs for HTM predictions
    # sdrs_test = []
    # for v in test_values:
    #    sdrs_test.append(encoder_htm.encode(v))

    sdr0 = sdrs_train[0]
    size = len(sdr0)
    dense_input = np.asarray(sdr0, dtype=np.int8)
    num_t = len(sdrs_train) * 10
    tm_outputs = []
    tm_prediction_masks = []
    sp = SpatialPooler(size, 200, 200)
    cells_per_column = 50
    tm = TemporalMemory(columns=sp.columns, cells_per_column=cells_per_column)

    tm_prediction_masks = [None] * len(sdrs_train)

    for t in range(num_t):
        idx = t % len(sdrs_train)
        sdr = sdrs_train[idx]

        dense_input = np.asarray(sdr, dtype=np.int8)
        mask, active_cols = sp.compute_active_columns(dense_input, inhibition_radius=1)

        tm_out = tm.step(active_cols)
        m = tm.get_predictive_columns_mask()

        tm_prediction_masks[idx] = m
        tm_outputs.append(tm_out)

    x_knn = []
    y_knn = []

    for idx, mask in enumerate(tm_prediction_masks):
        x_knn.append(mask.astype(np.int8))
        y_knn.append(train_values[idx])

    x_knn = np.asarray(x_knn)
    y_knn = np.asarray(y_knn)

    knn = KNeighborsRegressor(n_neighbors=1, weights="distance", metric="hamming")
    knn.fit(x_knn, y_knn)

    # try out the hot gym
    predictions = []
    actual_values = []
    # tm.reset_state()
    for t in range(100):
        idx = t % len(sdrs_train)
        sdr = sdrs_train[idx]

        dense_input = np.asarray(sdr, dtype=np.int8)
        mask, active_cols = sp.compute_active_columns(dense_input, inhibition_radius=10)

        tm.step(active_cols)
        m = tm.get_predictive_columns_mask()

        p = knn.predict(m.reshape(1, -1))[0]
        predictions.append(p)

        next_idx = (idx + 1) % len(sdrs_train)
        # target_col = input_data.columns[1]
        target_col = cols[0] if cols else None
        actual = None
        if target_col is not None and next_idx < len(input_records):
            actual = input_records[next_idx].get(target_col)
            try:
                actual = float(actual)
            except Exception:
                actual = None
        actual_values.append(actual)

    for pred, actual in zip(predictions, actual_values):
        print(f"Actual next step: {actual}, Prediction: {pred}")

    print("__actual_next_____predicted___error")
    y_true = []
    y_pred = []

    for i in range(len(predictions)):
        actual = actual_values[i]
        pred = predictions[i]

        y_true.append(actual)
        y_pred.append(pred)

        error = pred - actual
        print(f"{actual:9.3f}   {pred:9.3f}   {error:+9.3f}")
    """

    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
    print("RMSE:", rmse)
    """


if __name__ == "__main__":
    main()
