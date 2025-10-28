"""
Quick Test Script
Verify that everything is set up correctly
"""

import os
import sys


def print_status(message, status):
    """Print status with emoji"""
    emoji = " " if status else " "
    print(f"{emoji} {message}")


def check_file(filepath, description):
    """Check if file exists"""
    exists = os.path.exists(filepath)
    print_status(f"{description}: {filepath}", exists)
    return exists


def check_python_packages():
    """Check if required packages are installed"""
    print("\nChecking Python Packages...")
    packages = [
        'flask', 'pandas', 'numpy', 'sklearn',
        'yaml', 'xgboost', 'matplotlib', 'seaborn'
    ]

    all_installed = True

    for package in packages:
        try:
            __import__(package)
            print_status(f"{package}", True)
        except ImportError:
            print_status(f"{package}", False)
            all_installed = False

    return all_installed


def main():
    """Run all checks"""
    print("=" * 60)
    print("  DATA INSIGHT ML - QUICK TEST")
    print("=" * 60)

    # Check configuration files
    print("\nChecking Configuration Files...")
    config_ok = check_file('config.yaml', 'config.yaml')
    readme_ok = check_file('README.md', 'README.md')
    requirements_ok = check_file('requirements.txt', 'requirements.txt')

    # Check Python scripts
    print("\nChecking Python Scripts...")
    prepare_ok = check_file('prepare_data.py', 'prepare_data.py')
    train_ok = check_file('train_model.py', 'train_model.py')
    app_ok = check_file('app.py', 'app.py')

    # Check Python packages
    packages_ok = check_python_packages()

    # Check models directory
    print("\nChecking Models...")
    models_dir = os.path.exists('models')
    print_status(f"models/ directory", models_dir)

    if models_dir:
        model_ok = check_file('models/best_model.pkl', 'Trained model')
        scaler_ok = check_file('models/scaler.pkl', 'Scaler')
        features_ok = check_file('models/feature_columns.pkl', 'Features')

        if model_ok and scaler_ok and features_ok:
            print("\n Model is trained and ready!")
        else:
            print("\n Model not trained yet. Run: python train_model.py")
    else:
        print("\n Models directory not found. Run: python train_model.py")

    # Check data
    print("\nChecking Data...")
    data_ready = check_file('ml_ready_dataset.csv', 'Prepared dataset')

    if data_ready:
        print(" Data is prepared!")
    else:
        print(" Data not prepared yet. Run: python prepare_data.py")

    # Summary
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)

    all_ok = (config_ok and readme_ok and requirements_ok and
              prepare_ok and train_ok and app_ok and packages_ok)

    if all_ok:
        print("\n All core files present!")

        if data_ready:
            if models_dir and check_file('models/best_model.pkl', ''):
                print(" System is READY!")
                print("\n Next step:")
                print("  python app.py")
            else:
                print(" Data prepared, but model not trained.")
                print("\n Next step:")
                print("  python train_model.py")
        else:
            print(" Core files OK, but data not prepared.")
            print("\n Next step:")
            print("  python prepare_data.py your_data.csv")
    else:
        print("\n Some files are missing!")

        if not packages_ok:
            print("\n Install missing packages:")
            print("  pip install -r requirements.txt")

    print("\n" + "=" * 60)


if __name__ == '__main__':
    main()
