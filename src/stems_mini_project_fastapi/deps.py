"""FastAPI dependencies and global variables."""
import stems_mini_project as smp
import stems_mini_project_fastapi as smp_fapi


PRED_MODEL, DEVICE = smp.modeling.utils.load_model(
    smp_fapi.config.SETTINGS.PRED_MODEL_PATH,
    smp_fapi.config.SETTINGS.USE_CUDA,
    smp_fapi.config.SETTINGS.USE_MPS,
)
