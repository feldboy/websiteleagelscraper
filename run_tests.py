#!/usr/bin/env python3
"""
Test runner for the Multi-Agent Legal Research System.
Runs all tests and provides comprehensive test coverage reporting.
"""
import sys
import subprocess
import os
from pathlib import Path


def run_command(command, description):
    """Run a command and return success status."""
    print(f"\n🔄 {description}...")
    print(f"Command: {command}")
    
    try:
        result = subprocess.run(
            command.split(),
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent
        )
        
        if result.returncode == 0:
            print(f"✅ {description} - PASSED")
            if result.stdout:
                print(result.stdout)
            return True
        else:
            print(f"❌ {description} - FAILED")
            if result.stderr:
                print("STDERR:", result.stderr)
            if result.stdout:
                print("STDOUT:", result.stdout)
            return False
            
    except Exception as e:
        print(f"❌ {description} - ERROR: {e}")
        return False


def main():
    """Run all tests and quality checks."""
    print("🧪 Multi-Agent Legal Research System - Test Suite")
    print("=" * 60)
    
    # Check if we're in the right directory
    if not os.path.exists("tests") or not os.path.exists("agents"):
        print("❌ Error: Please run this script from the project root directory")
        sys.exit(1)
    
    # List of test commands to run
    test_commands = [
        # Code quality checks
        ("python3 -m black --check --diff agents tools config models tests", "Black code formatting check"),
        ("python3 -m isort --check-only --diff agents tools config models tests", "Import sorting check"),
        ("python3 -m flake8 agents tools config models tests --max-line-length=88 --extend-ignore=E203,W503", "Flake8 linting"),
        
        # Type checking (if mypy is available)
        ("python3 -m mypy agents tools config models --ignore-missing-imports", "Type checking with MyPy"),
        
        # Unit tests
        ("python3 -m pytest tests/test_agents/ -v", "Agent unit tests"),
        ("python3 -m pytest tests/test_tools/ -v", "Tools unit tests"),
        ("python3 -m pytest tests/test_models/ -v", "Models unit tests"),
        
        # Integration tests
        ("python3 -m pytest tests/test_integration/ -v", "Integration tests"),
        
        # Coverage report
        ("python3 -m pytest tests/ --cov=agents --cov=tools --cov=config --cov=models --cov-report=term-missing --cov-report=html", "Test coverage report"),
    ]
    
    passed = 0
    failed = 0
    
    for command, description in test_commands:
        if run_command(command, description):
            passed += 1
        else:
            failed += 1
    
    # Summary
    print("\n" + "=" * 60)
    print(f"📊 Test Summary:")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"📈 Total: {passed + failed}")
    
    if failed > 0:
        print(f"\n⚠️ {failed} test(s) failed. Please review the output above.")
        sys.exit(1)
    else:
        print(f"\n🎉 All tests passed! System is ready for deployment.")
        sys.exit(0)


if __name__ == "__main__":
    main()