"""Compliance and Regulatory Framework for Quantum Task Planner

Global compliance with GDPR, CCPA, PDPA and other data protection regulations.
"""

import hashlib
import json
import time
import uuid
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import logging
from pathlib import Path

from .quantum_i18n import LocalizationConfig, Region

logger = logging.getLogger(__name__)


class DataCategory(Enum):
    """Categories of data for compliance tracking."""
    PERSONAL_DATA = "personal"
    SENSITIVE_DATA = "sensitive"
    SYSTEM_METADATA = "system"
    PERFORMANCE_METRICS = "performance"
    TASK_EXECUTION = "task_execution"
    SECURITY_AUDIT = "security_audit"


class ProcessingPurpose(Enum):
    """Legal basis for data processing."""
    TASK_EXECUTION = "task_execution"
    PERFORMANCE_OPTIMIZATION = "performance"
    SECURITY_MONITORING = "security"
    SYSTEM_OPERATION = "operation"
    COMPLIANCE_MONITORING = "compliance"
    RESEARCH_DEVELOPMENT = "research"


class DataRetentionPolicy(Enum):
    """Data retention policies."""
    SHORT_TERM = "30_days"      # 30 days
    MEDIUM_TERM = "90_days"     # 90 days
    LONG_TERM = "365_days"      # 1 year
    EXTENDED = "2555_days"      # 7 years (compliance requirement)
    PERMANENT = "permanent"     # Never delete


@dataclass
class DataProcessingRecord:
    """Record of data processing activity for compliance."""
    record_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
    
    # Data subject information
    subject_id: Optional[str] = None
    data_category: DataCategory = DataCategory.SYSTEM_METADATA
    
    # Processing details
    purpose: ProcessingPurpose = ProcessingPurpose.SYSTEM_OPERATION
    legal_basis: str = "legitimate_interest"
    data_fields: List[str] = field(default_factory=list)
    
    # Retention and lifecycle
    retention_policy: DataRetentionPolicy = DataRetentionPolicy.MEDIUM_TERM
    scheduled_deletion: Optional[float] = None
    
    # Compliance tracking
    consent_id: Optional[str] = None
    consent_timestamp: Optional[float] = None
    processing_country: str = "US"
    
    def __post_init__(self):
        # Calculate scheduled deletion based on retention policy
        if self.scheduled_deletion is None:
            retention_days = {
                DataRetentionPolicy.SHORT_TERM: 30,
                DataRetentionPolicy.MEDIUM_TERM: 90, 
                DataRetentionPolicy.LONG_TERM: 365,
                DataRetentionPolicy.EXTENDED: 2555,
                DataRetentionPolicy.PERMANENT: None
            }
            
            days = retention_days.get(self.retention_policy)
            if days is not None:
                self.scheduled_deletion = self.timestamp + (days * 24 * 3600)
    
    def is_expired(self) -> bool:
        """Check if data should be deleted per retention policy."""
        if self.scheduled_deletion is None:
            return False
        return time.time() > self.scheduled_deletion
    
    def anonymize(self) -> None:
        """Anonymize personal data in record."""
        if self.subject_id:
            self.subject_id = hashlib.sha256(self.subject_id.encode()).hexdigest()[:16]
        
        # Remove or hash other identifying data
        anonymized_fields = []
        for field in self.data_fields:
            if any(keyword in field.lower() for keyword in ['name', 'email', 'id', 'user']):
                anonymized_fields.append(f"<anonymized_{field.split('_')[0]}>")
            else:
                anonymized_fields.append(field)
        
        self.data_fields = anonymized_fields


@dataclass
class ConsentRecord:
    """Record of user consent for data processing."""
    consent_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
    
    subject_id: str = ""
    consented_purposes: Set[ProcessingPurpose] = field(default_factory=set)
    consented_categories: Set[DataCategory] = field(default_factory=set)
    
    # Consent metadata
    consent_version: str = "1.0"
    consent_method: str = "explicit"  # explicit, implicit, legitimate_interest
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    
    # Consent lifecycle
    is_active: bool = True
    withdrawal_timestamp: Optional[float] = None
    
    def withdraw_consent(self) -> None:
        """Withdraw user consent."""
        self.is_active = False
        self.withdrawal_timestamp = time.time()
    
    def has_consent_for(self, purpose: ProcessingPurpose, category: DataCategory) -> bool:
        """Check if consent exists for specific purpose and data category."""
        return (self.is_active and 
                purpose in self.consented_purposes and 
                category in self.consented_categories)


class GDPRCompliance:
    """GDPR compliance implementation."""
    
    def __init__(self):
        self.processing_records: List[DataProcessingRecord] = []
        self.consent_records: Dict[str, ConsentRecord] = {}
        self.data_subject_requests: List[Dict[str, Any]] = []
    
    def record_processing_activity(self, record: DataProcessingRecord) -> None:
        """Record data processing activity under GDPR Article 30."""
        self.processing_records.append(record)
        logger.info(f"Recorded GDPR processing activity: {record.record_id}")
    
    def handle_subject_access_request(self, subject_id: str) -> Dict[str, Any]:
        """Handle GDPR Article 15 - Right of access."""
        subject_data = {
            "subject_id": subject_id,
            "request_timestamp": time.time(),
            "processing_records": [],
            "consent_records": [],
            "data_categories": set(),
            "processing_purposes": set()
        }
        
        # Find all processing records for subject
        for record in self.processing_records:
            if record.subject_id == subject_id:
                subject_data["processing_records"].append({
                    "record_id": record.record_id,
                    "timestamp": record.timestamp,
                    "purpose": record.purpose.value,
                    "legal_basis": record.legal_basis,
                    "data_fields": record.data_fields,
                    "retention_policy": record.retention_policy.value
                })
                subject_data["data_categories"].add(record.data_category.value)
                subject_data["processing_purposes"].add(record.purpose.value)
        
        # Find consent records
        if subject_id in self.consent_records:
            consent = self.consent_records[subject_id]
            subject_data["consent_records"].append({
                "consent_id": consent.consent_id,
                "timestamp": consent.timestamp,
                "consented_purposes": [p.value for p in consent.consented_purposes],
                "consented_categories": [c.value for c in consent.consented_categories],
                "is_active": consent.is_active,
                "withdrawal_timestamp": consent.withdrawal_timestamp
            })
        
        # Convert sets to lists for JSON serialization
        subject_data["data_categories"] = list(subject_data["data_categories"])
        subject_data["processing_purposes"] = list(subject_data["processing_purposes"])
        
        # Log the request
        self.data_subject_requests.append({
            "type": "access_request",
            "subject_id": subject_id,
            "timestamp": time.time(),
            "status": "completed"
        })
        
        return subject_data
    
    def handle_right_to_erasure(self, subject_id: str, reason: str = "withdrawal_of_consent") -> bool:
        """Handle GDPR Article 17 - Right to erasure (Right to be forgotten)."""
        deleted_records = 0
        
        # Remove or anonymize processing records
        for record in self.processing_records[:]:  # Copy list to modify during iteration
            if record.subject_id == subject_id:
                if reason in ["withdrawal_of_consent", "no_longer_necessary"]:
                    record.anonymize()
                    deleted_records += 1
                elif reason == "unlawful_processing":
                    self.processing_records.remove(record)
                    deleted_records += 1
        
        # Mark consent as withdrawn
        if subject_id in self.consent_records:
            self.consent_records[subject_id].withdraw_consent()
        
        # Log the erasure request
        self.data_subject_requests.append({
            "type": "erasure_request",
            "subject_id": subject_id,
            "timestamp": time.time(),
            "reason": reason,
            "records_affected": deleted_records,
            "status": "completed"
        })
        
        logger.info(f"Processed GDPR erasure request for {subject_id}: {deleted_records} records affected")
        return True
    
    def handle_data_portability_request(self, subject_id: str) -> Dict[str, Any]:
        """Handle GDPR Article 20 - Right to data portability."""
        portable_data = {
            "subject_id": subject_id,
            "export_timestamp": time.time(),
            "format": "JSON",
            "data": {}
        }
        
        # Export user's data in structured format
        for record in self.processing_records:
            if record.subject_id == subject_id:
                category = record.data_category.value
                if category not in portable_data["data"]:
                    portable_data["data"][category] = []
                
                portable_data["data"][category].append({
                    "timestamp": record.timestamp,
                    "purpose": record.purpose.value,
                    "data_fields": record.data_fields
                })
        
        # Log the portability request
        self.data_subject_requests.append({
            "type": "portability_request", 
            "subject_id": subject_id,
            "timestamp": time.time(),
            "status": "completed"
        })
        
        return portable_data
    
    def cleanup_expired_data(self) -> int:
        """Cleanup data that has exceeded retention periods."""
        expired_count = 0
        current_time = time.time()
        
        # Remove expired processing records
        for record in self.processing_records[:]:
            if record.is_expired():
                if record.data_category in [DataCategory.PERSONAL_DATA, DataCategory.SENSITIVE_DATA]:
                    record.anonymize()  # Anonymize instead of delete for audit trail
                else:
                    self.processing_records.remove(record)
                expired_count += 1
        
        logger.info(f"GDPR cleanup: processed {expired_count} expired records")
        return expired_count


class CCPACompliance:
    """CCPA compliance implementation."""
    
    def __init__(self):
        self.personal_info_records: List[DataProcessingRecord] = []
        self.consumer_requests: List[Dict[str, Any]] = []
        self.opt_out_requests: Set[str] = set()
    
    def record_personal_info_collection(self, record: DataProcessingRecord) -> None:
        """Record collection of personal information under CCPA."""
        if record.data_category in [DataCategory.PERSONAL_DATA, DataCategory.SENSITIVE_DATA]:
            self.personal_info_records.append(record)
            logger.info(f"Recorded CCPA personal info collection: {record.record_id}")
    
    def handle_right_to_know(self, consumer_id: str) -> Dict[str, Any]:
        """Handle CCPA Right to Know request."""
        disclosure = {
            "consumer_id": consumer_id,
            "request_timestamp": time.time(),
            "categories_collected": set(),
            "categories_sold": set(),
            "categories_disclosed": set(),
            "business_purposes": set(),
            "sources": set(),
            "third_parties": set()
        }
        
        for record in self.personal_info_records:
            if record.subject_id == consumer_id:
                disclosure["categories_collected"].add(record.data_category.value)
                disclosure["business_purposes"].add(record.purpose.value)
                disclosure["sources"].add("direct_collection")
        
        # Convert sets to lists
        for key in ["categories_collected", "categories_sold", "categories_disclosed", 
                   "business_purposes", "sources", "third_parties"]:
            disclosure[key] = list(disclosure[key])
        
        self.consumer_requests.append({
            "type": "right_to_know",
            "consumer_id": consumer_id,
            "timestamp": time.time(),
            "status": "completed"
        })
        
        return disclosure
    
    def handle_opt_out_request(self, consumer_id: str) -> bool:
        """Handle CCPA opt-out of sale request."""
        self.opt_out_requests.add(consumer_id)
        
        self.consumer_requests.append({
            "type": "opt_out_of_sale",
            "consumer_id": consumer_id, 
            "timestamp": time.time(),
            "status": "completed"
        })
        
        logger.info(f"Processed CCPA opt-out request for {consumer_id}")
        return True
    
    def is_opted_out(self, consumer_id: str) -> bool:
        """Check if consumer has opted out of sale."""
        return consumer_id in self.opt_out_requests


class PDPACompliance:
    """PDPA (Personal Data Protection Act) compliance for APAC region."""
    
    def __init__(self):
        self.personal_data_records: List[DataProcessingRecord] = []
        self.notification_logs: List[Dict[str, Any]] = []
        self.breach_records: List[Dict[str, Any]] = []
    
    def record_personal_data_processing(self, record: DataProcessingRecord) -> None:
        """Record personal data processing under PDPA."""
        if record.data_category in [DataCategory.PERSONAL_DATA, DataCategory.SENSITIVE_DATA]:
            self.personal_data_records.append(record)
    
    def handle_access_request(self, individual_id: str) -> Dict[str, Any]:
        """Handle PDPA individual access request."""
        individual_data = {
            "individual_id": individual_id,
            "request_timestamp": time.time(),
            "personal_data": [],
            "processing_purposes": set(),
            "disclosure_recipients": []
        }
        
        for record in self.personal_data_records:
            if record.subject_id == individual_id:
                individual_data["personal_data"].append({
                    "data_fields": record.data_fields,
                    "collection_timestamp": record.timestamp,
                    "purpose": record.purpose.value
                })
                individual_data["processing_purposes"].add(record.purpose.value)
        
        individual_data["processing_purposes"] = list(individual_data["processing_purposes"])
        return individual_data
    
    def notify_data_breach(self, breach_details: Dict[str, Any]) -> None:
        """Handle PDPA data breach notification requirements."""
        breach_record = {
            "breach_id": str(uuid.uuid4()),
            "timestamp": time.time(),
            "details": breach_details,
            "notification_status": "recorded",
            "authority_notified": False,
            "individuals_notified": False
        }
        
        self.breach_records.append(breach_record)
        
        # Log for compliance audit
        logger.critical(f"PDPA data breach recorded: {breach_record['breach_id']}")


class QuantumComplianceManager:
    """Main compliance management system for quantum task planner."""
    
    def __init__(self, config: LocalizationConfig):
        self.config = config
        
        # Initialize compliance systems based on region
        self.gdpr_compliance = GDPRCompliance() if config.gdpr_compliance else None
        self.ccpa_compliance = CCPACompliance() if config.ccpa_compliance else None  
        self.pdpa_compliance = PDPACompliance() if config.pdpa_compliance else None
        
        self.processing_records: List[DataProcessingRecord] = []
        self.consent_manager = ConsentManager()
        
        logger.info(f"Initialized compliance for region {config.region.value}")
    
    def record_data_processing(self, 
                             subject_id: Optional[str],
                             data_category: DataCategory,
                             purpose: ProcessingPurpose,
                             data_fields: List[str],
                             legal_basis: str = "legitimate_interest") -> str:
        """Record data processing activity across all applicable compliance frameworks."""
        
        record = DataProcessingRecord(
            subject_id=subject_id,
            data_category=data_category,
            purpose=purpose,
            data_fields=data_fields,
            legal_basis=legal_basis,
            processing_country=self._get_processing_country()
        )
        
        self.processing_records.append(record)
        
        # Record in applicable compliance systems
        if self.gdpr_compliance:
            self.gdpr_compliance.record_processing_activity(record)
        
        if self.ccpa_compliance:
            self.ccpa_compliance.record_personal_info_collection(record)
        
        if self.pdpa_compliance:
            self.pdpa_compliance.record_personal_data_processing(record)
        
        return record.record_id
    
    def handle_data_subject_request(self, request_type: str, subject_id: str, **kwargs) -> Dict[str, Any]:
        """Handle data subject rights requests across compliance frameworks."""
        
        results = {
            "request_type": request_type,
            "subject_id": subject_id,
            "timestamp": time.time(),
            "results": {}
        }
        
        if request_type == "access" and self.gdpr_compliance:
            results["results"]["gdpr"] = self.gdpr_compliance.handle_subject_access_request(subject_id)
        
        if request_type == "access" and self.ccpa_compliance:
            results["results"]["ccpa"] = self.ccpa_compliance.handle_right_to_know(subject_id)
        
        if request_type == "access" and self.pdpa_compliance:
            results["results"]["pdpa"] = self.pdpa_compliance.handle_access_request(subject_id)
        
        if request_type == "erasure" and self.gdpr_compliance:
            reason = kwargs.get("reason", "withdrawal_of_consent")
            results["results"]["gdpr"] = self.gdpr_compliance.handle_right_to_erasure(subject_id, reason)
        
        if request_type == "opt_out" and self.ccpa_compliance:
            results["results"]["ccpa"] = self.ccpa_compliance.handle_opt_out_request(subject_id)
        
        if request_type == "portability" and self.gdpr_compliance:
            results["results"]["gdpr"] = self.gdpr_compliance.handle_data_portability_request(subject_id)
        
        return results
    
    def perform_compliance_audit(self) -> Dict[str, Any]:
        """Perform comprehensive compliance audit."""
        audit_results = {
            "audit_timestamp": time.time(),
            "region": self.config.region.value,
            "total_processing_records": len(self.processing_records),
            "compliance_frameworks": [],
            "issues": [],
            "recommendations": []
        }
        
        # GDPR audit
        if self.gdpr_compliance:
            audit_results["compliance_frameworks"].append("GDPR")
            gdpr_audit = self._audit_gdpr_compliance()
            audit_results["gdpr_audit"] = gdpr_audit
            
            if gdpr_audit["issues"]:
                audit_results["issues"].extend(gdpr_audit["issues"])
        
        # CCPA audit  
        if self.ccpa_compliance:
            audit_results["compliance_frameworks"].append("CCPA")
            ccpa_audit = self._audit_ccpa_compliance()
            audit_results["ccpa_audit"] = ccpa_audit
            
            if ccpa_audit["issues"]:
                audit_results["issues"].extend(ccpa_audit["issues"])
        
        # PDPA audit
        if self.pdpa_compliance:
            audit_results["compliance_frameworks"].append("PDPA")
            pdpa_audit = self._audit_pdpa_compliance()
            audit_results["pdpa_audit"] = pdpa_audit
            
            if pdpa_audit["issues"]:
                audit_results["issues"].extend(pdpa_audit["issues"])
        
        # Generate recommendations
        audit_results["recommendations"] = self._generate_compliance_recommendations(audit_results)
        
        return audit_results
    
    def _audit_gdpr_compliance(self) -> Dict[str, Any]:
        """Audit GDPR compliance."""
        audit = {
            "processing_records": len(self.gdpr_compliance.processing_records),
            "consent_records": len(self.gdpr_compliance.consent_records),
            "subject_requests": len(self.gdpr_compliance.data_subject_requests),
            "issues": []
        }
        
        # Check for records without legal basis
        records_without_basis = [r for r in self.gdpr_compliance.processing_records 
                               if not r.legal_basis]
        if records_without_basis:
            audit["issues"].append(f"{len(records_without_basis)} records missing legal basis")
        
        # Check for expired data
        expired_records = [r for r in self.gdpr_compliance.processing_records if r.is_expired()]
        if expired_records:
            audit["issues"].append(f"{len(expired_records)} records exceeded retention period")
        
        return audit
    
    def _audit_ccpa_compliance(self) -> Dict[str, Any]:
        """Audit CCPA compliance."""
        audit = {
            "personal_info_records": len(self.ccpa_compliance.personal_info_records),
            "consumer_requests": len(self.ccpa_compliance.consumer_requests),
            "opt_out_requests": len(self.ccpa_compliance.opt_out_requests),
            "issues": []
        }
        
        # Check for proper disclosures
        categories_collected = set(r.data_category for r in self.ccpa_compliance.personal_info_records)
        if len(categories_collected) > 0 and not hasattr(self, 'privacy_policy_updated'):
            audit["issues"].append("Privacy policy may need updating for collected categories")
        
        return audit
    
    def _audit_pdpa_compliance(self) -> Dict[str, Any]:
        """Audit PDPA compliance.""" 
        audit = {
            "personal_data_records": len(self.pdpa_compliance.personal_data_records),
            "breach_records": len(self.pdpa_compliance.breach_records),
            "issues": []
        }
        
        # Check for unresolved breaches
        unresolved_breaches = [b for b in self.pdpa_compliance.breach_records 
                              if not b["authority_notified"]]
        if unresolved_breaches:
            audit["issues"].append(f"{len(unresolved_breaches)} breaches require authority notification")
        
        return audit
    
    def _generate_compliance_recommendations(self, audit_results: Dict[str, Any]) -> List[str]:
        """Generate compliance recommendations based on audit."""
        recommendations = []
        
        if audit_results["issues"]:
            recommendations.append("Address identified compliance issues immediately")
        
        # Data minimization
        if audit_results["total_processing_records"] > 1000:
            recommendations.append("Consider data minimization strategies")
        
        # Privacy by design
        recommendations.append("Implement privacy by design principles in new features")
        
        # Regular audits
        recommendations.append("Schedule regular compliance audits (quarterly recommended)")
        
        return recommendations
    
    def cleanup_expired_data(self) -> Dict[str, int]:
        """Cleanup expired data across all compliance frameworks."""
        cleanup_results = {}
        
        if self.gdpr_compliance:
            cleanup_results["gdpr"] = self.gdpr_compliance.cleanup_expired_data()
        
        # Add CCPA and PDPA cleanup as needed
        cleanup_results["total_records"] = len(self.processing_records)
        
        return cleanup_results
    
    def _get_processing_country(self) -> str:
        """Get processing country based on region."""
        country_mapping = {
            Region.NORTH_AMERICA: "US",
            Region.EUROPE: "DE",  # Default to Germany for GDPR
            Region.ASIA_PACIFIC: "SG",  # Default to Singapore for PDPA
            Region.LATIN_AMERICA: "BR",
            Region.MIDDLE_EAST_AFRICA: "AE"
        }
        
        return country_mapping.get(self.config.region, "US")
    
    def export_compliance_report(self, output_path: str) -> None:
        """Export comprehensive compliance report."""
        report = self.perform_compliance_audit()
        
        try:
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            logger.info(f"Compliance report exported to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to export compliance report: {e}")


class ConsentManager:
    """Manage user consent across compliance frameworks."""
    
    def __init__(self):
        self.consent_records: Dict[str, ConsentRecord] = {}
    
    def record_consent(self, subject_id: str, 
                      purposes: List[ProcessingPurpose],
                      categories: List[DataCategory],
                      method: str = "explicit") -> str:
        """Record user consent."""
        
        consent = ConsentRecord(
            subject_id=subject_id,
            consented_purposes=set(purposes),
            consented_categories=set(categories),
            consent_method=method
        )
        
        self.consent_records[subject_id] = consent
        
        logger.info(f"Recorded consent for {subject_id}: {len(purposes)} purposes, {len(categories)} categories")
        
        return consent.consent_id
    
    def check_consent(self, subject_id: str, purpose: ProcessingPurpose, category: DataCategory) -> bool:
        """Check if user has given consent for specific processing."""
        
        if subject_id not in self.consent_records:
            return False
        
        consent = self.consent_records[subject_id]
        return consent.has_consent_for(purpose, category)
    
    def withdraw_consent(self, subject_id: str) -> bool:
        """Withdraw user consent."""
        
        if subject_id in self.consent_records:
            self.consent_records[subject_id].withdraw_consent()
            logger.info(f"Consent withdrawn for {subject_id}")
            return True
        
        return False