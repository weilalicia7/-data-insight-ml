"""
Domain Manager for Multi-Domain ML System
Manages multiple pre-trained models for different use cases
"""

import os
import json
import pickle
import pandas as pd
import numpy as np
from typing import Dict, List, Optional


class DomainManager:
    """
    Manages multiple ML domains with pre-trained models.

    Each domain represents a specific use case (e.g., donor retention,
    grant scoring) with its own model, features, and example data.
    """

    def __init__(self, domains_dir='domains'):
        """
        Initialize the Domain Manager.

        Args:
            domains_dir: Directory containing domain folders
        """
        self.domains_dir = domains_dir
        self.available_domains = {}
        self.current_domain = None
        self.current_model_data = None

        # Scan for available domains
        if os.path.exists(domains_dir):
            self.scan_domains()

    def scan_domains(self):
        """Scan and catalog all available domains"""
        if not os.path.exists(self.domains_dir):
            print(f"⚠ Domains directory not found: {self.domains_dir}")
            return

        for domain_name in os.listdir(self.domains_dir):
            domain_path = os.path.join(self.domains_dir, domain_name)

            if os.path.isdir(domain_path):
                metadata_path = os.path.join(domain_path, 'metadata.json')

                # Load metadata if exists
                metadata = {}
                if os.path.exists(metadata_path):
                    try:
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)
                    except Exception as e:
                        print(f"⚠ Error loading metadata for {domain_name}: {e}")
                        continue

                # Check for required files
                has_model = os.path.exists(os.path.join(domain_path, 'model.pkl'))
                has_scaler = os.path.exists(os.path.join(domain_path, 'scaler.pkl'))
                has_features = os.path.exists(os.path.join(domain_path, 'features.pkl'))
                has_data = os.path.exists(os.path.join(domain_path, 'example_data.csv'))

                self.available_domains[domain_name] = {
                    'path': domain_path,
                    'metadata': metadata,
                    'has_model': has_model,
                    'has_scaler': has_scaler,
                    'has_features': has_features,
                    'has_data': has_data,
                    'ready': has_model and has_scaler and has_features
                }

        print(f"[OK] Found {len(self.available_domains)} domains")

    def list_domains(self) -> List[Dict]:
        """
        Get list of available domains.

        Returns:
            List of domain information dictionaries
        """
        domains_list = []

        for name, info in self.available_domains.items():
            metadata = info.get('metadata', {})

            domains_list.append({
                'name': name,
                'display_name': metadata.get('display_name', name.replace('_', ' ').title()),
                'description': metadata.get('description', 'No description available'),
                'icon': metadata.get('icon', '📊'),
                'use_case': metadata.get('use_case', ''),
                'accuracy': metadata.get('accuracy', 0),
                'ready': info['ready'],
                'has_demo_data': info['has_data']
            })

        return sorted(domains_list, key=lambda x: x['display_name'])

    def load_domain(self, domain_name: str) -> Dict:
        """
        Load a specific domain's model and configuration.

        Args:
            domain_name: Name of the domain to load

        Returns:
            Dictionary containing model, scaler, features, and metadata

        Raises:
            ValueError: If domain doesn't exist or is not ready
        """
        if domain_name not in self.available_domains:
            raise ValueError(f"Domain '{domain_name}' not found")

        domain_info = self.available_domains[domain_name]

        if not domain_info['ready']:
            raise ValueError(f"Domain '{domain_name}' is not ready (missing files)")

        domain_path = domain_info['path']

        # Load model
        model_path = os.path.join(domain_path, 'model.pkl')
        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        # Load scaler
        scaler_path = os.path.join(domain_path, 'scaler.pkl')
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)

        # Load features
        features_path = os.path.join(domain_path, 'features.pkl')
        with open(features_path, 'rb') as f:
            features = pickle.load(f)

        self.current_domain = domain_name
        self.current_model_data = {
            'model': model,
            'scaler': scaler,
            'features': features,
            'metadata': domain_info['metadata']
        }

        print(f"[OK] Loaded domain: {domain_name}")

        return self.current_model_data

    def get_example_data(self, domain_name: str, n_rows: int = 10) -> List[Dict]:
        """
        Get example data for a domain.

        Args:
            domain_name: Name of the domain
            n_rows: Number of rows to return

        Returns:
            List of example records as dictionaries
        """
        if domain_name not in self.available_domains:
            raise ValueError(f"Domain '{domain_name}' not found")

        domain_path = self.available_domains[domain_name]['path']
        data_path = os.path.join(domain_path, 'example_data.csv')

        if not os.path.exists(data_path):
            return []

        df = pd.read_csv(data_path)
        return df.head(n_rows).to_dict('records')

    def get_sample_predictions(self, domain_name: str) -> List[Dict]:
        """
        Get sample prediction examples from metadata.

        Args:
            domain_name: Name of the domain

        Returns:
            List of sample input/output examples
        """
        if domain_name not in self.available_domains:
            return []

        metadata = self.available_domains[domain_name].get('metadata', {})
        return metadata.get('sample_inputs', [])

    def predict(self, input_data: Dict) -> Dict:
        """
        Make prediction using currently loaded domain.

        Args:
            input_data: Dictionary of feature values

        Returns:
            Dictionary containing prediction, confidence, and recommendations

        Raises:
            ValueError: If no domain is loaded
        """
        if self.current_model_data is None:
            raise ValueError("No domain loaded. Call load_domain() first.")

        model = self.current_model_data['model']
        scaler = self.current_model_data['scaler']
        features = self.current_model_data['features']
        metadata = self.current_model_data['metadata']

        # Ensure all features are present
        feature_values = []
        for feature in features:
            feature_values.append(input_data.get(feature, 0))

        # Convert to numpy array
        X = np.array([feature_values])

        # Scale if scaler is available
        if scaler is not None:
            X_scaled = scaler.transform(X)
        else:
            X_scaled = X

        # Make prediction
        prediction = model.predict(X_scaled)[0]

        # Get probability if available
        confidence = None
        probabilities = None

        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(X_scaled)[0]
            probabilities = proba.tolist()
            confidence = float(max(proba))

        # Get recommendation based on prediction
        recommendations = metadata.get('recommendations', {})
        threshold = recommendations.get('high_confidence_threshold', 0.7)

        if confidence and confidence >= threshold:
            if prediction == 1:
                recommendation = recommendations.get('action_positive', 'Positive outcome predicted')
            else:
                recommendation = recommendations.get('action_negative', 'Negative outcome predicted')
        else:
            recommendation = 'Manual review recommended (low confidence)'

        return {
            'prediction': int(prediction),
            'confidence': confidence,
            'probabilities': probabilities,
            'recommendation': recommendation,
            'domain': self.current_domain
        }

    def get_domain_info(self, domain_name: str) -> Dict:
        """
        Get detailed information about a domain.

        Args:
            domain_name: Name of the domain

        Returns:
            Dictionary with domain information
        """
        if domain_name not in self.available_domains:
            raise ValueError(f"Domain '{domain_name}' not found")

        domain_info = self.available_domains[domain_name]
        metadata = domain_info.get('metadata', {})

        return {
            'name': domain_name,
            'display_name': metadata.get('display_name', domain_name),
            'description': metadata.get('description', ''),
            'use_case': metadata.get('use_case', ''),
            'icon': metadata.get('icon', '📊'),
            'target_variable': metadata.get('target_variable', ''),
            'target_description': metadata.get('target_description', ''),
            'features_count': metadata.get('features_count', 0),
            'example_rows': metadata.get('example_rows', 0),
            'model_type': metadata.get('model_type', 'Unknown'),
            'accuracy': metadata.get('accuracy', 0),
            'precision': metadata.get('precision', 0),
            'recall': metadata.get('recall', 0),
            'f1_score': metadata.get('f1_score', 0),
            'training_date': metadata.get('training_date', ''),
            'ready': domain_info['ready'],
            'has_demo_data': domain_info['has_data']
        }


# Convenience functions for quick use

def list_available_domains(domains_dir='domains') -> List[Dict]:
    """Quick function to list all available domains"""
    dm = DomainManager(domains_dir)
    return dm.list_domains()


def quick_predict(domain_name: str, input_data: Dict, domains_dir='domains') -> Dict:
    """
    Quick prediction without managing DomainManager instance.

    Args:
        domain_name: Name of the domain
        input_data: Input features
        domains_dir: Path to domains directory

    Returns:
        Prediction result
    """
    dm = DomainManager(domains_dir)
    dm.load_domain(domain_name)
    return dm.predict(input_data)


if __name__ == '__main__':
    # Test the domain manager
    print("=" * 60)
    print("Domain Manager Test")
    print("=" * 60)

    dm = DomainManager()

    print("\nAvailable Domains:")
    domains = dm.list_domains()

    for domain in domains:
        status = "[Ready]" if domain['ready'] else "[Not Ready]"
        print(f"  {domain['display_name']}: {status}")
        print(f"     {domain['description']}")

    if domains:
        # Test loading first available domain
        first_domain = domains[0]['name']
        print(f"\nTesting domain: {first_domain}")

        try:
            dm.load_domain(first_domain)
            print(f"[OK] Successfully loaded {first_domain}")

            # Get domain info
            info = dm.get_domain_info(first_domain)
            print(f"  Model: {info['model_type']}")
            print(f"  Accuracy: {info['accuracy']:.2%}")
            print(f"  Features: {info['features_count']}")

        except Exception as e:
            print(f"[ERROR] {e}")
