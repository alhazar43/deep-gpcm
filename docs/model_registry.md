# Model Registry and Color System

## Overview

The Deep-GPCM project now uses a centralized model registry system that defines model metadata including colors, display names, and descriptions. This ensures consistency across all components of the system.

## Model Registry

Located in `models/factory.py`, the registry defines:

```python
MODEL_REGISTRY = {
    'deep_gpcm': {
        'class': DeepGPCM,
        'color': '#ff7f0e',  # Orange
        'display_name': 'Deep-GPCM',
        'description': 'Deep learning GPCM with DKVMN memory'
    },
    'attn_gpcm': {
        'class': EnhancedAttentionGPCM,
        'color': '#1f77b4',  # Blue
        'display_name': 'Attention-GPCM',
        'description': 'Attention-enhanced GPCM with multi-head attention'
    },
    'coral_gpcm_proper': {
        'class': CORALGPCM,
        'color': '#e377c2',  # Pink
        'display_name': 'CORAL-GPCM',
        'description': 'CORAL-enhanced GPCM with ordinal regression'
    }
}
```

## Usage

### Creating Models

Models are created using the factory with metadata automatically attached:

```python
from models import create_model

# Create model with metadata
model = create_model('deep_gpcm', n_questions=100, n_cats=4)

# Access metadata
print(model.model_color)    # '#ff7f0e'
print(model.display_name)   # 'Deep-GPCM'
print(model.description)    # 'Deep learning GPCM with DKVMN memory'
```

### Getting Model Information

Use utility functions to access model metadata:

```python
from models.factory import get_model_color, get_model_metadata, get_all_model_types

# Get color for a model
color = get_model_color('deep_gpcm')  # '#ff7f0e'

# Get all metadata
metadata = get_model_metadata('attn_gpcm')
# {'color': '#1f77b4', 'display_name': 'Attention-GPCM', ...}

# List all available models
models = get_all_model_types()  # ['deep_gpcm', 'attn_gpcm', 'coral_gpcm_proper']
```

### Plotting Integration

The plotting system automatically uses factory-defined colors:

```python
# In plot_metrics.py and irt_analysis.py
def get_model_color(self, model_name: str) -> str:
    # First tries factory colors
    from models.factory import get_model_color as factory_get_color
    return factory_get_color(model_name)
    # Falls back to local mapping for backward compatibility
```

## Benefits

1. **Centralized Management**: Single source of truth for model metadata
2. **Consistency**: Same colors and names used everywhere
3. **Extensibility**: Easy to add new models with their metadata
4. **Type Safety**: Models have typed metadata attributes
5. **Backward Compatibility**: Fallback mechanisms for legacy code

## Adding New Models

To add a new model:

1. Add entry to `MODEL_REGISTRY`:
```python
'new_model': {
    'class': NewModelClass,
    'color': '#hexcolor',
    'display_name': 'Display Name',
    'description': 'Model description'
}
```

2. Add model-specific parameters in `create_model()` function
3. The model will automatically get consistent colors in all plots

## Color Scheme

Current models use distinct colors from the tab10 colormap:
- **Deep-GPCM**: Orange (#ff7f0e) - Base DKVMN model
- **Attention-GPCM**: Blue (#1f77b4) - Attention-enhanced
- **CORAL-GPCM**: Pink (#e377c2) - Ordinal regression

Unknown models default to gray (#808080).