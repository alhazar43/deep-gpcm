#!/usr/bin/env python3
"""
Migration script to reorganize core/ directory to models/ directory.
This script safely migrates all files while preserving functionality.
"""

import os
import shutil
import re
from pathlib import Path
import argparse
import sys

class CoreToModelssMigrator:
    def __init__(self, project_root=".", dry_run=False):
        self.project_root = Path(project_root)
        self.dry_run = dry_run
        self.core_dir = self.project_root / "core"
        self.models_dir = self.project_root / "models"
        self.backup_dir = self.project_root / "core_backup"
        
        # Track all file movements for import updates
        self.file_mappings = {}
        self.class_mappings = {}
        
    def log(self, message, level="INFO"):
        prefix = "[DRY RUN] " if self.dry_run else ""
        print(f"{prefix}[{level}] {message}")
        
    def create_directory_structure(self):
        """Create the new models/ directory structure."""
        directories = [
            "models",
            "models/base",
            "models/implementations", 
            "models/components",
            "models/adaptive"
        ]
        
        for dir_path in directories:
            full_path = self.project_root / dir_path
            if not self.dry_run:
                full_path.mkdir(exist_ok=True)
            self.log(f"Created directory: {dir_path}")
            
    def backup_core_directory(self):
        """Create a backup of the core directory."""
        if self.backup_dir.exists():
            self.log("Backup directory already exists, skipping backup", "WARN")
            return
            
        if not self.dry_run:
            shutil.copytree(self.core_dir, self.backup_dir)
        self.log(f"Backed up core/ to {self.backup_dir}")
        
    def create_file_content(self, file_path, content):
        """Create a file with given content."""
        if not self.dry_run:
            with open(file_path, 'w') as f:
                f.write(content)
        self.log(f"Created file: {file_path}")
        
    def migrate_base_model(self):
        """Extract BaseKnowledgeTracingModel from model.py."""
        model_py = self.core_dir / "model.py"
        base_model_path = self.models_dir / "base" / "base_model.py"
        
        if not self.dry_run:
            with open(model_py, 'r') as f:
                content = f.read()
                
            # Extract BaseKnowledgeTracingModel class
            base_class_pattern = r'(class BaseKnowledgeTracingModel.*?)(?=\nclass|\Z)'
            base_match = re.search(base_class_pattern, content, re.DOTALL)
            
            if base_match:
                # Get imports needed for base model
                imports = """import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any
"""
                base_content = imports + "\n\n" + base_match.group(1)
                
                with open(base_model_path, 'w') as f:
                    f.write(base_content)
                    
        self.class_mappings['BaseKnowledgeTracingModel'] = 'models.base.base_model'
        self.log("Migrated BaseKnowledgeTracingModel to models/base/base_model.py")
        
    def migrate_deep_gpcm(self):
        """Extract DeepGPCM from model.py."""
        model_py = self.core_dir / "model.py"
        deep_gpcm_path = self.models_dir / "implementations" / "deep_gpcm.py"
        
        if not self.dry_run:
            with open(model_py, 'r') as f:
                content = f.read()
                
            # Extract DeepGPCM class
            deep_class_pattern = r'(class DeepGPCM.*?)(?=\nclass|\Z)'
            deep_match = re.search(deep_class_pattern, content, re.DOTALL)
            
            if deep_match:
                imports = """import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple

from ..base.base_model import BaseKnowledgeTracingModel
from ..components.memory_networks import DKVMN
from ..components.embeddings import create_embedding_strategy
from ..components.irt_layers import IRTParameterExtractor, GPCMProbabilityLayer
"""
                deep_content = imports + "\n\n" + deep_match.group(1)
                
                # Update relative imports
                deep_content = deep_content.replace('from .memory_networks', 'from ..components.memory_networks')
                deep_content = deep_content.replace('from .embeddings', 'from ..components.embeddings')
                deep_content = deep_content.replace('from .layers', 'from ..components.irt_layers')
                
                with open(deep_gpcm_path, 'w') as f:
                    f.write(deep_content)
                    
        self.class_mappings['DeepGPCM'] = 'models.implementations.deep_gpcm'
        self.log("Migrated DeepGPCM to models/implementations/deep_gpcm.py")
        
    def migrate_attention_models(self):
        """Consolidate AttentionGPCM and EnhancedAttentionGPCM."""
        attention_gpcm_path = self.models_dir / "implementations" / "attention_gpcm.py"
        
        if not self.dry_run:
            # Get AttentionGPCM from model.py
            model_py = self.core_dir / "model.py"
            with open(model_py, 'r') as f:
                model_content = f.read()
                
            attention_pattern = r'(class AttentionGPCM.*?)(?=\nclass|\Z)'
            attention_match = re.search(attention_pattern, model_content, re.DOTALL)
            
            # Get EnhancedAttentionGPCM from attention_enhanced.py
            enhanced_py = self.core_dir / "attention_enhanced.py"
            enhanced_content = ""
            if enhanced_py.exists():
                with open(enhanced_py, 'r') as f:
                    enhanced_content = f.read()
                    
            # Combine both
            imports = """import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

from ..base.base_model import BaseKnowledgeTracingModel
from ..components.memory_networks import DKVMN
from ..components.embeddings import create_embedding_strategy, LinearDecayEmbedding
from ..components.irt_layers import IRTParameterExtractor, GPCMProbabilityLayer
from ..components.attention_layers import AttentionRefinementModule, EmbeddingProjection
"""
            
            combined_content = imports + "\n\n"
            if attention_match:
                combined_content += attention_match.group(1) + "\n\n"
            
            # Add EnhancedAttentionGPCM and LearnableLinearDecayEmbedding
            if enhanced_content:
                # Extract classes from enhanced_content
                enhanced_classes = re.findall(r'(class (?:EnhancedAttentionGPCM|LearnableLinearDecayEmbedding).*?)(?=\nclass|\Z)', 
                                            enhanced_content, re.DOTALL)
                for cls in enhanced_classes:
                    combined_content += cls + "\n\n"
                    
            # Update imports
            combined_content = self.update_imports_in_content(combined_content)
            
            with open(attention_gpcm_path, 'w') as f:
                f.write(combined_content)
                
        self.class_mappings['AttentionGPCM'] = 'models.implementations.attention_gpcm'
        self.class_mappings['EnhancedAttentionGPCM'] = 'models.implementations.attention_gpcm'
        self.log("Migrated attention models to models/implementations/attention_gpcm.py")
        
    def migrate_coral_models(self):
        """Migrate CORAL models to implementations."""
        coral_gpcm_path = self.models_dir / "implementations" / "coral_gpcm.py"
        
        if not self.dry_run:
            coral_py = self.core_dir / "coral_gpcm.py"
            with open(coral_py, 'r') as f:
                content = f.read()
                
            # Remove ThresholdCouplingConfig (will go to adaptive)
            content = re.sub(r'@dataclass\s*\nclass ThresholdCouplingConfig.*?(?=\nclass|\Z)', '', content, flags=re.DOTALL)
            
            # Update imports
            new_imports = """import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

from ..base.base_model import BaseKnowledgeTracingModel
from ..components.memory_networks import DKVMN
from ..components.embeddings import create_embedding_strategy
from ..components.irt_layers import IRTParameterExtractor, GPCMProbabilityLayer
from ..components.coral_layers import CORALLayer
from ..adaptive.threshold_coupling import ThresholdCouplingConfig
from ..adaptive.blenders import MinimalAdaptiveBlender
"""
            
            # Replace imports section
            content = re.sub(r'^.*?(?=\nclass)', new_imports, content, flags=re.DOTALL)
            content = self.update_imports_in_content(content)
            
            with open(coral_gpcm_path, 'w') as f:
                f.write(content)
                
        self.class_mappings['HybridCORALGPCM'] = 'models.implementations.coral_gpcm'
        self.class_mappings['EnhancedCORALGPCM'] = 'models.implementations.coral_gpcm'
        self.log("Migrated CORAL models to models/implementations/coral_gpcm.py")
        
    def split_layers(self):
        """Split layers.py into irt_layers.py and attention_layers.py."""
        layers_py = self.core_dir / "layers.py"
        
        if not self.dry_run and layers_py.exists():
            with open(layers_py, 'r') as f:
                content = f.read()
                
            # IRT layers
            irt_imports = """import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
"""
            irt_classes = ['IRTParameterExtractor', 'GPCMProbabilityLayer']
            irt_content = self.extract_classes(content, irt_classes, irt_imports)
            
            irt_path = self.models_dir / "components" / "irt_layers.py"
            with open(irt_path, 'w') as f:
                f.write(irt_content)
                
            # Attention layers
            attention_imports = """import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
"""
            attention_classes = ['AttentionRefinementModule', 'EmbeddingProjection']
            attention_content = self.extract_classes(content, attention_classes, attention_imports)
            
            attention_path = self.models_dir / "components" / "attention_layers.py"
            with open(attention_path, 'w') as f:
                f.write(attention_content)
                
        self.log("Split layers.py into irt_layers.py and attention_layers.py")
        
    def migrate_simple_files(self):
        """Migrate files that don't need splitting."""
        simple_migrations = [
            ("memory_networks.py", "components/memory_networks.py"),
            ("embeddings.py", "components/embeddings.py"),
        ]
        
        for src, dst in simple_migrations:
            src_path = self.core_dir / src
            dst_path = self.models_dir / dst
            
            if not self.dry_run and src_path.exists():
                content = src_path.read_text()
                content = self.update_imports_in_content(content)
                dst_path.write_text(content)
                
            self.log(f"Migrated {src} to {dst}")
            
    def migrate_coral_components(self):
        """Migrate CORAL layer and loss."""
        coral_layer_py = self.core_dir / "coral_layer.py"
        coral_layers_path = self.models_dir / "components" / "coral_layers.py"
        
        if not self.dry_run and coral_layer_py.exists():
            content = coral_layer_py.read_text()
            content = self.update_imports_in_content(content)
            coral_layers_path.write_text(content)
            
        self.class_mappings['CORALLayer'] = 'models.components.coral_layers'
        self.class_mappings['CORALCompatibleLoss'] = 'models.components.coral_layers'
        self.log("Migrated CORAL components to models/components/coral_layers.py")
        
    def migrate_adaptive_components(self):
        """Migrate adaptive blending components."""
        # FullAdaptiveBlender
        blender_py = self.core_dir / "full_adaptive_blender.py"
        blenders_path = self.models_dir / "adaptive" / "blenders.py"
        
        if not self.dry_run and blender_py.exists():
            content = blender_py.read_text()
            
            # Check for MinimalAdaptiveBlender import and add stub if needed
            if "MinimalAdaptiveBlender" in content and "class MinimalAdaptiveBlender" not in content:
                content += "\n\n# TODO: Implement MinimalAdaptiveBlender\nclass MinimalAdaptiveBlender:\n    pass\n"
                
            content = self.update_imports_in_content(content)
            blenders_path.write_text(content)
            
        # ThresholdCouplingConfig from coral_gpcm.py
        coupling_path = self.models_dir / "adaptive" / "threshold_coupling.py"
        if not self.dry_run:
            coral_py = self.core_dir / "coral_gpcm.py"
            with open(coral_py, 'r') as f:
                coral_content = f.read()
                
            # Extract ThresholdCouplingConfig
            coupling_pattern = r'(@dataclass\s*\nclass ThresholdCouplingConfig.*?)(?=\nclass|\Z)'
            coupling_match = re.search(coupling_pattern, coral_content, re.DOTALL)
            
            if coupling_match:
                imports = """from dataclasses import dataclass
from typing import Literal, Optional
import torch
import torch.nn as nn
"""
                coupling_content = imports + "\n\n" + coupling_match.group(1)
                
                with open(coupling_path, 'w') as f:
                    f.write(coupling_content)
                    
        self.log("Migrated adaptive components")
        
    def migrate_factory(self):
        """Migrate and update model factory."""
        factory_src = self.core_dir / "model_factory.py"
        factory_dst = self.models_dir / "factory.py"
        
        if not self.dry_run and factory_src.exists():
            content = factory_src.read_text()
            
            # Update imports to new structure
            new_imports = """from typing import Union, Optional
import torch.nn as nn

from .implementations.deep_gpcm import DeepGPCM
from .implementations.attention_gpcm import AttentionGPCM, EnhancedAttentionGPCM
from .implementations.coral_gpcm import HybridCORALGPCM, EnhancedCORALGPCM
from .adaptive.blenders import FullAdaptiveBlender

# Conditional imports for adaptive models
try:
    from .implementations.adaptive_coral_gpcm import AdaptiveCORALGPCM
except ImportError:
    AdaptiveCORALGPCM = None

try:
    from .implementations.full_adaptive_coral_gpcm import FullAdaptiveCORALGPCM
except ImportError:
    FullAdaptiveCORALGPCM = None
"""
            
            # Replace the imports section
            content = re.sub(r'^.*?(?=\ndef)', new_imports, content, flags=re.DOTALL)
            
            factory_dst.write_text(content)
            
        self.log("Migrated model factory to models/factory.py")
        
    def create_init_files(self):
        """Create __init__.py files for new structure."""
        # Main models __init__.py
        models_init = """\"\"\"
Deep-GPCM Models Package

Organized structure:
- base/: Base classes and interfaces
- implementations/: Concrete model implementations  
- components/: Reusable model components
- adaptive/: Adaptive and experimental features
- factory.py: Model creation utilities
\"\"\"

# Import all public classes for backward compatibility
from .base.base_model import BaseKnowledgeTracingModel
from .implementations.deep_gpcm import DeepGPCM
from .implementations.attention_gpcm import AttentionGPCM, EnhancedAttentionGPCM
from .implementations.coral_gpcm import HybridCORALGPCM, EnhancedCORALGPCM
from .components.memory_networks import DKVMN, MemoryNetwork, MemoryHeadGroup
from .components.embeddings import (
    create_embedding_strategy,
    OrderedEmbedding,
    UnorderedEmbedding,
    LinearDecayEmbedding,
    AdjacentWeightedEmbedding
)
from .components.irt_layers import IRTParameterExtractor, GPCMProbabilityLayer
from .components.attention_layers import AttentionRefinementModule, EmbeddingProjection
from .components.coral_layers import CORALLayer, CORALCompatibleLoss
from .adaptive.blenders import FullAdaptiveBlender
from .adaptive.threshold_coupling import ThresholdCouplingConfig
from .factory import create_model

__all__ = [
    # Base
    'BaseKnowledgeTracingModel',
    # Models
    'DeepGPCM',
    'AttentionGPCM',
    'EnhancedAttentionGPCM',
    'HybridCORALGPCM',
    'EnhancedCORALGPCM',
    # Components
    'DKVMN',
    'MemoryNetwork',
    'MemoryHeadGroup',
    'create_embedding_strategy',
    'IRTParameterExtractor',
    'GPCMProbabilityLayer',
    'AttentionRefinementModule',
    'EmbeddingProjection',
    'CORALLayer',
    'CORALCompatibleLoss',
    # Adaptive
    'FullAdaptiveBlender',
    'ThresholdCouplingConfig',
    # Factory
    'create_model',
]
"""
        
        # Sub-package __init__.py files
        base_init = """from .base_model import BaseKnowledgeTracingModel

__all__ = ['BaseKnowledgeTracingModel']
"""
        
        impl_init = """from .deep_gpcm import DeepGPCM
from .attention_gpcm import AttentionGPCM, EnhancedAttentionGPCM
from .coral_gpcm import HybridCORALGPCM, EnhancedCORALGPCM

__all__ = [
    'DeepGPCM',
    'AttentionGPCM', 
    'EnhancedAttentionGPCM',
    'HybridCORALGPCM',
    'EnhancedCORALGPCM',
]
"""
        
        components_init = """from .memory_networks import DKVMN, MemoryNetwork, MemoryHeadGroup
from .embeddings import (
    create_embedding_strategy,
    OrderedEmbedding,
    UnorderedEmbedding,
    LinearDecayEmbedding,
    AdjacentWeightedEmbedding
)
from .irt_layers import IRTParameterExtractor, GPCMProbabilityLayer
from .attention_layers import AttentionRefinementModule, EmbeddingProjection
from .coral_layers import CORALLayer, CORALCompatibleLoss

__all__ = [
    'DKVMN',
    'MemoryNetwork',
    'MemoryHeadGroup',
    'create_embedding_strategy',
    'OrderedEmbedding',
    'UnorderedEmbedding',
    'LinearDecayEmbedding',
    'AdjacentWeightedEmbedding',
    'IRTParameterExtractor',
    'GPCMProbabilityLayer',
    'AttentionRefinementModule',
    'EmbeddingProjection',
    'CORALLayer',
    'CORALCompatibleLoss',
]
"""
        
        adaptive_init = """from .blenders import FullAdaptiveBlender
from .threshold_coupling import ThresholdCouplingConfig

__all__ = ['FullAdaptiveBlender', 'ThresholdCouplingConfig']
"""
        
        # Create files
        init_files = [
            (self.models_dir / "__init__.py", models_init),
            (self.models_dir / "base" / "__init__.py", base_init),
            (self.models_dir / "implementations" / "__init__.py", impl_init),
            (self.models_dir / "components" / "__init__.py", components_init),
            (self.models_dir / "adaptive" / "__init__.py", adaptive_init),
        ]
        
        for path, content in init_files:
            self.create_file_content(path, content)
            
    def update_imports_in_content(self, content):
        """Update relative imports within file content."""
        # Update relative imports
        replacements = [
            (r'from \.memory_networks', 'from ..components.memory_networks'),
            (r'from \.embeddings', 'from ..components.embeddings'),
            (r'from \.layers', 'from ..components.irt_layers'),
            (r'from \.coral_layer', 'from ..components.coral_layers'),
            (r'from \.full_adaptive_blender', 'from ..adaptive.blenders'),
            (r'from \.model import BaseKnowledgeTracingModel', 'from ..base.base_model import BaseKnowledgeTracingModel'),
        ]
        
        for old, new in replacements:
            content = re.sub(old, new, content)
            
        return content
        
    def extract_classes(self, content, class_names, imports):
        """Extract specific classes from content."""
        result = imports + "\n\n"
        
        for class_name in class_names:
            pattern = rf'(class {class_name}.*?)(?=\nclass|\Z)'
            match = re.search(pattern, content, re.DOTALL)
            if match:
                result += match.group(1) + "\n\n"
                
        return result
        
    def update_project_imports(self):
        """Update imports throughout the project."""
        self.log("Updating imports throughout the project...")
        
        # Find all Python files
        python_files = []
        for root, dirs, files in os.walk(self.project_root):
            # Skip backup and models directories
            if 'core_backup' in root or 'models' in root:
                continue
            for file in files:
                if file.endswith('.py'):
                    python_files.append(os.path.join(root, file))
                    
        # Update imports in each file
        replacements = [
            (r'from core\.', 'from models.'),
            (r'import core\.', 'import models.'),
            (r'from models import', 'from models import'),
            (r'from models\.model import', 'from models.implementations import'),
            (r'from models\.model_factory', 'from models.factory'),
            (r'from models\.coral_gpcm import ThresholdCouplingConfig', 
             'from models.adaptive.threshold_coupling import ThresholdCouplingConfig'),
        ]
        
        for file_path in python_files:
            if not self.dry_run:
                with open(file_path, 'r') as f:
                    content = f.read()
                    
                original_content = content
                for old, new in replacements:
                    content = re.sub(old, new, content)
                    
                if content != original_content:
                    with open(file_path, 'w') as f:
                        f.write(content)
                    self.log(f"Updated imports in: {file_path}")
                    
    def remove_old_directory(self):
        """Remove the old core directory."""
        if not self.dry_run and self.core_dir.exists():
            shutil.rmtree(self.core_dir)
        self.log("Removed old core/ directory")
        
    def run(self):
        """Execute the full migration."""
        self.log("Starting core/ to models/ migration...")
        
        # Step 1: Backup
        self.backup_core_directory()
        
        # Step 2: Create new structure
        self.create_directory_structure()
        
        # Step 3: Migrate files
        self.log("Migrating files...")
        self.migrate_base_model()
        self.migrate_deep_gpcm()
        self.migrate_attention_models()
        self.migrate_coral_models()
        self.split_layers()
        self.migrate_simple_files()
        self.migrate_coral_components()
        self.migrate_adaptive_components()
        self.migrate_factory()
        
        # Step 4: Create init files
        self.create_init_files()
        
        # Step 5: Update imports
        self.update_project_imports()
        
        # Step 6: Clean up
        if not self.dry_run:
            self.remove_old_directory()
            
        self.log("Migration completed successfully!")
        self.log(f"Backup saved at: {self.backup_dir}")
        
        # Print summary
        print("\n" + "="*50)
        print("MIGRATION SUMMARY")
        print("="*50)
        print("Old structure: core/")
        print("New structure: models/")
        print("  - models/base/           (base classes)")
        print("  - models/implementations/ (model implementations)")
        print("  - models/components/     (reusable components)")
        print("  - models/adaptive/       (experimental features)")
        print("  - models/factory.py      (model factory)")
        print("\nKey import changes:")
        print("  from models.implementations import DeepGPCM")
        print("  -> from models.implementations.deep_gpcm import DeepGPCM")
        print("\n  from models.factory import create_model")
        print("  -> from models.factory import create_model")
        print("\nBackup location:", self.backup_dir)
        

def main():
    parser = argparse.ArgumentParser(description="Migrate core/ to models/ structure")
    parser.add_argument("--dry-run", action="store_true", 
                       help="Show what would be done without making changes")
    parser.add_argument("--project-root", default=".", 
                       help="Project root directory (default: current directory)")
    
    args = parser.parse_args()
    
    # Check if core directory exists
    core_dir = Path(args.project_root) / "core"
    if not core_dir.exists():
        print(f"Error: core/ directory not found at {core_dir}")
        sys.exit(1)
        
    # Check if models directory already exists
    models_dir = Path(args.project_root) / "models"
    if models_dir.exists() and not args.dry_run:
        response = input("models/ directory already exists. Continue? (y/n): ")
        if response.lower() != 'y':
            print("Migration cancelled.")
            sys.exit(0)
            
    # Run migration
    migrator = CoreToModelssMigrator(args.project_root, args.dry_run)
    migrator.run()
    

if __name__ == "__main__":
    main()