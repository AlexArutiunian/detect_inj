import os
try:
    import keras
    m = keras.saving.load_model("artifacts/outputs/run/kaggle/working/output_gru_run/model.keras")
    m.export("artifacts/outputs/run/kaggle/working/output_gru_run/model_saved")  # экспорт в TF SavedModel (директория)
    print("Exported to model_saved/")
except Exception as e:
    raise SystemExit("Нужен keras>=3 для чтения .keras: " + repr(e))