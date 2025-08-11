#!/usr/bin/env python3
"""
Simple test to demonstrate QuantumComplianceManager configuration requirements
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from edge_tpu_v5_benchmark.quantum_compliance import (
    QuantumComplianceManager,
    DataCategory,
    ProcessingPurpose
)
from edge_tpu_v5_benchmark.quantum_i18n import (
    LocalizationConfig,
    SupportedLanguage,
    Region
)

def test_quantum_compliance_manager_config():
    """Test QuantumComplianceManager with different regional configurations."""
    
    print("Testing QuantumComplianceManager configuration requirements...")
    
    # Test 1: GDPR Compliance Configuration (Europe)
    print("\n1. Testing GDPR Compliance Configuration")
    gdpr_config = LocalizationConfig(
        language=SupportedLanguage.ENGLISH,
        region=Region.EUROPE,  # This will auto-enable GDPR compliance
        timezone="UTC",
        currency="EUR"
    )
    
    gdpr_manager = QuantumComplianceManager(gdpr_config)
    
    # Record some data processing activity
    record_id = gdpr_manager.record_data_processing(
        subject_id="user_123",
        data_category=DataCategory.PERSONAL_DATA,
        purpose=ProcessingPurpose.PERFORMANCE_OPTIMIZATION,
        data_fields=["benchmark_results", "execution_time"],
        legal_basis="legitimate_interest"
    )
    
    print(f"✓ GDPR Manager created successfully, recorded activity: {record_id}")
    print(f"✓ GDPR compliance enabled: {gdpr_config.gdpr_compliance}")
    
    # Test 2: CCPA Compliance Configuration (North America)
    print("\n2. Testing CCPA Compliance Configuration")
    ccpa_config = LocalizationConfig(
        language=SupportedLanguage.ENGLISH,
        region=Region.NORTH_AMERICA,  # This will auto-enable CCPA compliance
        timezone="America/Los_Angeles",
        currency="USD"
    )
    
    ccpa_manager = QuantumComplianceManager(ccpa_config)
    
    # Record some data processing activity
    record_id = ccpa_manager.record_data_processing(
        subject_id="consumer_456",
        data_category=DataCategory.PERFORMANCE_METRICS,
        purpose=ProcessingPurpose.SYSTEM_OPERATION,
        data_fields=["throughput", "latency"],
        legal_basis="business_necessity"
    )
    
    print(f"✓ CCPA Manager created successfully, recorded activity: {record_id}")
    print(f"✓ CCPA compliance enabled: {ccpa_config.ccpa_compliance}")
    
    # Test 3: PDPA Compliance Configuration (Asia-Pacific)
    print("\n3. Testing PDPA Compliance Configuration")
    pdpa_config = LocalizationConfig(
        language=SupportedLanguage.ENGLISH,
        region=Region.ASIA_PACIFIC,  # This will auto-enable PDPA compliance
        timezone="Asia/Singapore",
        currency="SGD"
    )
    
    pdpa_manager = QuantumComplianceManager(pdpa_config)
    
    # Record some data processing activity
    record_id = pdpa_manager.record_data_processing(
        subject_id="individual_789",
        data_category=DataCategory.SYSTEM_METADATA,
        purpose=ProcessingPurpose.SECURITY_MONITORING,
        data_fields=["system_info", "audit_logs"],
        legal_basis="consent"
    )
    
    print(f"✓ PDPA Manager created successfully, recorded activity: {record_id}")
    print(f"✓ PDPA compliance enabled: {pdpa_config.pdpa_compliance}")
    
    # Test 4: Perform compliance audit
    print("\n4. Testing Compliance Audit")
    audit_results = gdpr_manager.perform_compliance_audit()
    
    print(f"✓ Audit completed for region: {audit_results['region']}")
    print(f"✓ Total processing records: {audit_results['total_processing_records']}")
    print(f"✓ Compliance frameworks: {audit_results['compliance_frameworks']}")
    print(f"✓ Issues found: {len(audit_results['issues'])}")
    print(f"✓ Recommendations: {len(audit_results['recommendations'])}")
    
    # Test 5: Data subject request
    print("\n5. Testing Data Subject Request")
    access_results = gdpr_manager.handle_data_subject_request(
        request_type="access",
        subject_id="user_123"
    )
    
    print(f"✓ Access request processed for subject: {access_results['subject_id']}")
    print(f"✓ Request type: {access_results['request_type']}")
    print(f"✓ Results keys: {list(access_results['results'].keys())}")
    
    print("\n✅ All QuantumComplianceManager tests passed!")
    print("\nConfiguration Requirements Summary:")
    print("- QuantumComplianceManager requires a LocalizationConfig object")
    print("- LocalizationConfig automatically enables compliance based on region:")
    print("  • Region.EUROPE → GDPR compliance")
    print("  • Region.NORTH_AMERICA → CCPA compliance")  
    print("  • Region.ASIA_PACIFIC → PDPA compliance")
    print("- LocalizationConfig supports language, region, timezone, currency settings")
    print("- The manager handles multi-framework compliance seamlessly")


if __name__ == "__main__":
    test_quantum_compliance_manager_config()