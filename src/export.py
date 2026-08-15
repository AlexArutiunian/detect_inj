import os
try:
    import keras
    m = keras.saving.load_model("outputs_run/kaggle/working/output_gru_run/model.keras")
    m.export("outputs_run/kaggle/working/output_gru_run/model_saved")  # экспорт в TF SavedModel (директория)
    print("Exported to model_saved/")
except Exception as e:
    raise SystemExit("Нужен keras>=3 для чтения .keras: " + repr(e))