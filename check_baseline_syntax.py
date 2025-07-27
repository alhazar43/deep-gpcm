#!/usr/bin/env python3
"""
Syntax check for baseline model without running torch code.
"""

import sys
import ast

def check_baseline_syntax():
    """Check baseline model syntax."""
    try:
        with open('models/baseline.py', 'r') as f:
            content = f.read()
        
        # Parse the AST to check syntax
        tree = ast.parse(content)
        print("✅ Baseline model syntax is valid")
        
        # Count classes and functions
        classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
        functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        
        print(f"   Classes: {len(classes)}")
        print(f"   Functions: {len(functions)}")
        
        # List main classes
        class_names = [cls.name for cls in classes]
        print(f"   Class names: {class_names}")
        
        return True
    except SyntaxError as e:
        print(f"❌ Syntax error in baseline.py: {e}")
        return False
    except Exception as e:
        print(f"❌ Error checking baseline.py: {e}")
        return False

def check_akvmn_syntax():
    """Check AKVMN model syntax."""
    try:
        with open('models/akvmn.py', 'r') as f:
            content = f.read()
        
        # Parse the AST to check syntax
        tree = ast.parse(content)
        print("✅ AKVMN model syntax is valid")
        
        # Count classes and functions
        classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
        functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        
        print(f"   Classes: {len(classes)}")
        print(f"   Functions: {len(functions)}")
        
        # List main classes
        class_names = [cls.name for cls in classes]
        print(f"   Class names: {class_names}")
        
        return True
    except SyntaxError as e:
        print(f"❌ Syntax error in akvmn.py: {e}")
        return False
    except Exception as e:
        print(f"❌ Error checking akvmn.py: {e}")
        return False

def main():
    """Main function."""
    print("=== Model Syntax Check ===")
    
    baseline_ok = check_baseline_syntax()
    print()
    akvmn_ok = check_akvmn_syntax()
    
    print(f"\n=== Summary ===")
    print(f"Baseline syntax: {'✅' if baseline_ok else '❌'}")
    print(f"AKVMN syntax: {'✅' if akvmn_ok else '❌'}")

if __name__ == "__main__":
    main()