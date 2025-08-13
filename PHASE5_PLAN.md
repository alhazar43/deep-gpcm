Phase 5: Architectural Improvements - Directory Structure Reorganization

This branch implements:
1. Consolidated directory structure: results/dataset_name/{metrics,plots,irt_plots,models}/
2. Configuration consolidation and import standardization
3. Backward compatibility through PathManager legacy mode

Target structure:
- Current: results/{plots,irt_plots,test,train,validation}/dataset/
- New: results/dataset/{metrics,plots,irt_plots,models}/

Implementation order:
1. Critical: path_utils.py, train.py, evaluate.py, main.py
2. High: analysis/irt_analysis.py, migrate_structure.py  
3. Moderate: config/ module consolidation
4. Low: import standardization across remaining files
