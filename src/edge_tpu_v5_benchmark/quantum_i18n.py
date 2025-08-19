"""Internationalization and Localization for Quantum Task Planner

Global-first implementation with support for multiple languages and regions.
"""

import json
import locale
import logging
import os
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class SupportedLanguage(Enum):
    """Supported languages for internationalization."""
    ENGLISH = "en"
    SPANISH = "es"
    FRENCH = "fr"
    GERMAN = "de"
    JAPANESE = "ja"
    CHINESE_SIMPLIFIED = "zh-CN"
    CHINESE_TRADITIONAL = "zh-TW"
    KOREAN = "ko"
    PORTUGUESE = "pt"
    ITALIAN = "it"
    RUSSIAN = "ru"
    ARABIC = "ar"


class Region(Enum):
    """Supported regions for localization."""
    NORTH_AMERICA = "NA"
    EUROPE = "EU"
    ASIA_PACIFIC = "APAC"
    LATIN_AMERICA = "LATAM"
    MIDDLE_EAST_AFRICA = "MEA"


@dataclass
class LocalizationConfig:
    """Configuration for localization settings."""
    language: SupportedLanguage = SupportedLanguage.ENGLISH
    region: Region = Region.NORTH_AMERICA
    timezone: str = "UTC"
    date_format: str = "%Y-%m-%d"
    time_format: str = "%H:%M:%S"
    number_format: str = "decimal"  # decimal, scientific, engineering
    currency: str = "USD"

    # Regional compliance settings
    gdpr_compliance: bool = False
    ccpa_compliance: bool = False
    pdpa_compliance: bool = False

    def __post_init__(self):
        # Auto-configure compliance based on region
        if self.region == Region.EUROPE:
            self.gdpr_compliance = True
        elif self.region == Region.NORTH_AMERICA:
            self.ccpa_compliance = True
        elif self.region == Region.ASIA_PACIFIC:
            self.pdpa_compliance = True


class QuantumMessageCatalog:
    """Message catalog for quantum system internationalization."""

    def __init__(self, language: SupportedLanguage = SupportedLanguage.ENGLISH):
        self.language = language
        self.messages: Dict[str, Dict[str, str]] = {}
        self._load_default_messages()

    def _load_default_messages(self) -> None:
        """Load default message translations."""
        self.messages = {
            # Core quantum messages
            "quantum.task.created": {
                "en": "Quantum task '{name}' created successfully",
                "es": "Tarea cuántica '{name}' creada exitosamente",
                "fr": "Tâche quantique '{name}' créée avec succès",
                "de": "Quantenaufgabe '{name}' erfolgreich erstellt",
                "ja": "量子タスク '{name}' が正常に作成されました",
                "zh-CN": "量子任务 '{name}' 创建成功",
                "ko": "양자 작업 '{name}'이 성공적으로 생성되었습니다",
                "pt": "Tarefa quântica '{name}' criada com sucesso",
                "it": "Attività quantistica '{name}' creata con successo",
                "ru": "Квантовая задача '{name}' успешно создана",
                "ar": "تم إنشاء المهمة الكمية '{name}' بنجاح"
            },

            "quantum.task.executing": {
                "en": "Executing quantum task '{name}'",
                "es": "Ejecutando tarea cuántica '{name}'",
                "fr": "Exécution de la tâche quantique '{name}'",
                "de": "Ausführung der Quantenaufgabe '{name}'",
                "ja": "量子タスク '{name}' を実行中",
                "zh-CN": "正在执行量子任务 '{name}'",
                "ko": "양자 작업 '{name}' 실행 중",
                "pt": "Executando tarefa quântica '{name}'",
                "it": "Esecuzione dell'attività quantistica '{name}'",
                "ru": "Выполнение квантовой задачи '{name}'",
                "ar": "تنفيذ المهمة الكمية '{name}'"
            },

            "quantum.task.completed": {
                "en": "Quantum task '{name}' completed successfully",
                "es": "Tarea cuántica '{name}' completada exitosamente",
                "fr": "Tâche quantique '{name}' terminée avec succès",
                "de": "Quantenaufgabe '{name}' erfolgreich abgeschlossen",
                "ja": "量子タスク '{name}' が正常に完了しました",
                "zh-CN": "量子任务 '{name}' 成功完成",
                "ko": "양자 작업 '{name}'이 성공적으로 완료되었습니다",
                "pt": "Tarefa quântica '{name}' concluída com sucesso",
                "it": "Attività quantistica '{name}' completata con successo",
                "ru": "Квантовая задача '{name}' успешно выполнена",
                "ar": "اكتملت المهمة الكمية '{name}' بنجاح"
            },

            "quantum.task.failed": {
                "en": "Quantum task '{name}' failed: {error}",
                "es": "Tarea cuántica '{name}' falló: {error}",
                "fr": "Échec de la tâche quantique '{name}': {error}",
                "de": "Quantenaufgabe '{name}' fehlgeschlagen: {error}",
                "ja": "量子タスク '{name}' が失敗しました: {error}",
                "zh-CN": "量子任务 '{name}' 失败: {error}",
                "ko": "양자 작업 '{name}' 실패: {error}",
                "pt": "Tarefa quântica '{name}' falhou: {error}",
                "it": "Attività quantistica '{name}' fallita: {error}",
                "ru": "Квантовая задача '{name}' не выполнена: {error}",
                "ar": "فشلت المهمة الكمية '{name}': {error}"
            },

            # Quantum states
            "quantum.state.superposition": {
                "en": "Superposition",
                "es": "Superposición",
                "fr": "Superposition",
                "de": "Superposition",
                "ja": "重ね合わせ",
                "zh-CN": "叠加态",
                "ko": "중첩",
                "pt": "Superposição",
                "it": "Sovrapposizione",
                "ru": "Суперпозиция",
                "ar": "التداخل"
            },

            "quantum.state.entangled": {
                "en": "Entangled",
                "es": "Entrelazado",
                "fr": "Intriqué",
                "de": "Verschränkt",
                "ja": "もつれ",
                "zh-CN": "纠缠",
                "ko": "얽힘",
                "pt": "Emaranhado",
                "it": "Intrecciato",
                "ru": "Запутанный",
                "ar": "متشابك"
            },

            "quantum.state.collapsed": {
                "en": "Collapsed",
                "es": "Colapsado",
                "fr": "Effondré",
                "de": "Kollabiert",
                "ja": "崩壊",
                "zh-CN": "坍缩",
                "ko": "붕괴",
                "pt": "Colapsado",
                "it": "Collassato",
                "ru": "Коллапсированный",
                "ar": "منهار"
            },

            # Performance metrics
            "metrics.throughput": {
                "en": "Throughput",
                "es": "Rendimiento",
                "fr": "Débit",
                "de": "Durchsatz",
                "ja": "スループット",
                "zh-CN": "吞吐量",
                "ko": "처리량",
                "pt": "Taxa de transferência",
                "it": "Throughput",
                "ru": "Пропускная способность",
                "ar": "الإنتاجية"
            },

            "metrics.latency": {
                "en": "Latency",
                "es": "Latencia",
                "fr": "Latence",
                "de": "Latenz",
                "ja": "レイテンシ",
                "zh-CN": "延迟",
                "ko": "지연시간",
                "pt": "Latência",
                "it": "Latenza",
                "ru": "Задержка",
                "ar": "زمن الاستجابة"
            },

            # Error messages
            "error.validation.failed": {
                "en": "Validation failed: {details}",
                "es": "Validación fallida: {details}",
                "fr": "Échec de la validation: {details}",
                "de": "Validierung fehlgeschlagen: {details}",
                "ja": "検証に失敗しました: {details}",
                "zh-CN": "验证失败: {details}",
                "ko": "검증 실패: {details}",
                "pt": "Validação falhou: {details}",
                "it": "Validazione fallita: {details}",
                "ru": "Проверка не пройдена: {details}",
                "ar": "فشل في التحقق: {details}"
            },

            "error.resource.unavailable": {
                "en": "Resource '{resource}' is unavailable",
                "es": "El recurso '{resource}' no está disponible",
                "fr": "La ressource '{resource}' n'est pas disponible",
                "de": "Ressource '{resource}' ist nicht verfügbar",
                "ja": "リソース '{resource}' は利用できません",
                "zh-CN": "资源 '{resource}' 不可用",
                "ko": "리소스 '{resource}'를 사용할 수 없습니다",
                "pt": "Recurso '{resource}' não está disponível",
                "it": "Risorsa '{resource}' non disponibile",
                "ru": "Ресурс '{resource}' недоступен",
                "ar": "المورد '{resource}' غير متاح"
            },

            # Security messages
            "security.access.denied": {
                "en": "Access denied: insufficient permissions",
                "es": "Acceso denegado: permisos insuficientes",
                "fr": "Accès refusé: permissions insuffisantes",
                "de": "Zugriff verweigert: unzureichende Berechtigungen",
                "ja": "アクセス拒否: 権限が不十分です",
                "zh-CN": "访问被拒绝: 权限不足",
                "ko": "액세스 거부: 권한이 부족합니다",
                "pt": "Acesso negado: permissões insuficientes",
                "it": "Accesso negato: permessi insufficienti",
                "ru": "Доступ запрещен: недостаточно прав",
                "ar": "تم رفض الوصول: أذونات غير كافية"
            },

            # Compliance messages
            "compliance.gdpr.notice": {
                "en": "This system complies with GDPR data protection regulations",
                "es": "Este sistema cumple con las regulaciones de protección de datos GDPR",
                "fr": "Ce système est conforme aux réglementations RGPD de protection des données",
                "de": "Dieses System entspricht den DSGVO-Datenschutzbestimmungen",
                "ja": "このシステムはGDPRデータ保護規則に準拠しています",
                "zh-CN": "此系统符合GDPR数据保护法规",
                "ko": "이 시스템은 GDPR 데이터 보호 규정을 준수합니다",
                "pt": "Este sistema está em conformidade com os regulamentos de proteção de dados GDPR",
                "it": "Questo sistema è conforme alle normative GDPR sulla protezione dei dati",
                "ru": "Эта система соответствует требованиям GDPR по защите данных",
                "ar": "يمتثل هذا النظام للوائح حماية البيانات GDPR"
            }
        }

    def get_message(self, key: str, **kwargs) -> str:
        """Get localized message with optional formatting."""
        lang_code = self.language.value

        if key not in self.messages:
            logger.warning(f"Message key '{key}' not found in catalog")
            return key

        message_dict = self.messages[key]

        # Try requested language, fall back to English
        if lang_code in message_dict:
            message = message_dict[lang_code]
        elif "en" in message_dict:
            message = message_dict["en"]
            logger.warning(f"Language '{lang_code}' not found for key '{key}', using English")
        else:
            logger.error(f"No translations found for key '{key}'")
            return key

        # Format message with provided kwargs
        try:
            return message.format(**kwargs)
        except KeyError as e:
            logger.error(f"Missing format parameter {e} for message key '{key}'")
            return message
        except Exception as e:
            logger.error(f"Error formatting message '{key}': {e}")
            return message

    def add_message(self, key: str, translations: Dict[str, str]) -> None:
        """Add custom message translations."""
        self.messages[key] = translations
        logger.info(f"Added message key '{key}' with {len(translations)} translations")

    def load_from_file(self, file_path: str) -> None:
        """Load messages from JSON file."""
        try:
            with open(file_path, encoding='utf-8') as f:
                file_messages = json.load(f)
                self.messages.update(file_messages)
                logger.info(f"Loaded messages from {file_path}")
        except Exception as e:
            logger.error(f"Failed to load messages from {file_path}: {e}")


class NumberFormatter:
    """Number formatting for different locales."""

    def __init__(self, config: LocalizationConfig):
        self.config = config

    def format_number(self, value: float, precision: int = 2) -> str:
        """Format number according to locale."""
        if self.config.number_format == "scientific":
            return f"{value:.{precision}e}"
        elif self.config.number_format == "engineering":
            # Engineering notation (powers of 1000)
            if value == 0:
                return "0"

            exponent = int(value and (value != 0) and (3 * (len(str(int(abs(value)))) - 1) // 3))
            mantissa = value / (10 ** exponent)
            return f"{mantissa:.{precision}f}×10^{exponent}"
        else:
            # Decimal format with locale-specific separators
            if self.config.language in [SupportedLanguage.FRENCH, SupportedLanguage.GERMAN]:
                # European format: comma for decimal, space for thousands
                formatted = f"{value:,.{precision}f}".replace(",", " ").replace(".", ",")
                return formatted.replace(" ", " ")  # Use thin space
            else:
                # Anglo format: period for decimal, comma for thousands
                return f"{value:,.{precision}f}"

    def format_currency(self, value: float, currency: Optional[str] = None) -> str:
        """Format currency according to locale."""
        currency = currency or self.config.currency
        formatted_number = self.format_number(value, 2)

        currency_symbols = {
            "USD": "$", "EUR": "€", "GBP": "£", "JPY": "¥",
            "CNY": "¥", "KRW": "₩", "RUB": "₽", "BRL": "R$"
        }

        symbol = currency_symbols.get(currency, currency)

        # Currency placement varies by locale
        if currency in ["EUR"] and self.config.language in [SupportedLanguage.GERMAN, SupportedLanguage.FRENCH]:
            return f"{formatted_number} {symbol}"
        else:
            return f"{symbol}{formatted_number}"

    def format_percentage(self, value: float, precision: int = 1) -> str:
        """Format percentage according to locale."""
        formatted = self.format_number(value * 100, precision)

        # Some locales put space before %
        if self.config.language in [SupportedLanguage.FRENCH]:
            return f"{formatted} %"
        else:
            return f"{formatted}%"


class DateTimeFormatter:
    """Date and time formatting for different locales."""

    def __init__(self, config: LocalizationConfig):
        self.config = config

    def format_date(self, timestamp: float) -> str:
        """Format date according to locale."""
        import datetime
        dt = datetime.datetime.fromtimestamp(timestamp)

        # Locale-specific date formats
        formats = {
            SupportedLanguage.ENGLISH: "%Y-%m-%d",
            SupportedLanguage.GERMAN: "%d.%m.%Y",
            SupportedLanguage.FRENCH: "%d/%m/%Y",
            SupportedLanguage.JAPANESE: "%Y年%m月%d日",
            SupportedLanguage.CHINESE_SIMPLIFIED: "%Y年%m月%d日",
            SupportedLanguage.KOREAN: "%Y년 %m월 %d일",
        }

        date_format = formats.get(self.config.language, self.config.date_format)
        return dt.strftime(date_format)

    def format_time(self, timestamp: float) -> str:
        """Format time according to locale."""
        import datetime
        dt = datetime.datetime.fromtimestamp(timestamp)

        # 24-hour vs 12-hour format based on region
        if self.config.region == Region.NORTH_AMERICA:
            time_format = "%I:%M:%S %p"  # 12-hour with AM/PM
        else:
            time_format = "%H:%M:%S"  # 24-hour format

        return dt.strftime(time_format)

    def format_duration(self, seconds: float) -> str:
        """Format duration in human-readable form."""
        if seconds < 60:
            return f"{seconds:.2f}s"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            remaining_seconds = seconds % 60
            return f"{minutes}m {remaining_seconds:.1f}s"
        else:
            hours = int(seconds // 3600)
            remaining_minutes = int((seconds % 3600) // 60)
            return f"{hours}h {remaining_minutes}m"


class QuantumLocalizer:
    """Main localization system for quantum task planner."""

    def __init__(self, config: Optional[LocalizationConfig] = None):
        self.config = config or LocalizationConfig()
        self.message_catalog = QuantumMessageCatalog(self.config.language)
        self.number_formatter = NumberFormatter(self.config)
        self.datetime_formatter = DateTimeFormatter(self.config)

        # Set up logging
        self.logger = logging.getLogger(f"quantum_i18n.{self.config.language.value}")

    def t(self, key: str, **kwargs) -> str:
        """Translate message (short alias for get_message)."""
        return self.message_catalog.get_message(key, **kwargs)

    def get_message(self, key: str, **kwargs) -> str:
        """Get localized message."""
        return self.message_catalog.get_message(key, **kwargs)

    def format_number(self, value: float, precision: int = 2) -> str:
        """Format number according to locale."""
        return self.number_formatter.format_number(value, precision)

    def format_currency(self, value: float, currency: Optional[str] = None) -> str:
        """Format currency according to locale."""
        return self.number_formatter.format_currency(value, currency)

    def format_percentage(self, value: float, precision: int = 1) -> str:
        """Format percentage according to locale."""
        return self.number_formatter.format_percentage(value, precision)

    def format_date(self, timestamp: float) -> str:
        """Format date according to locale."""
        return self.datetime_formatter.format_date(timestamp)

    def format_time(self, timestamp: float) -> str:
        """Format time according to locale."""
        return self.datetime_formatter.format_time(timestamp)

    def format_duration(self, seconds: float) -> str:
        """Format duration in human-readable form."""
        return self.datetime_formatter.format_duration(seconds)

    def get_compliance_notice(self) -> List[str]:
        """Get compliance notices for current configuration."""
        notices = []

        if self.config.gdpr_compliance:
            notices.append(self.get_message("compliance.gdpr.notice"))

        if self.config.ccpa_compliance:
            notices.append("This system complies with CCPA privacy regulations")

        if self.config.pdpa_compliance:
            notices.append("This system complies with PDPA data protection requirements")

        return notices

    def detect_locale_from_environment(self) -> LocalizationConfig:
        """Detect locale settings from environment."""
        try:
            # Try to get system locale
            system_locale = locale.getdefaultlocale()
            lang_code = system_locale[0] if system_locale[0] else "en_US"

            # Parse language and region
            if "_" in lang_code:
                lang_part, region_part = lang_code.split("_", 1)
            else:
                lang_part = lang_code
                region_part = "US"

            # Map to supported languages
            lang_mapping = {
                "en": SupportedLanguage.ENGLISH,
                "es": SupportedLanguage.SPANISH,
                "fr": SupportedLanguage.FRENCH,
                "de": SupportedLanguage.GERMAN,
                "ja": SupportedLanguage.JAPANESE,
                "zh": SupportedLanguage.CHINESE_SIMPLIFIED,
                "ko": SupportedLanguage.KOREAN,
                "pt": SupportedLanguage.PORTUGUESE,
                "it": SupportedLanguage.ITALIAN,
                "ru": SupportedLanguage.RUSSIAN,
                "ar": SupportedLanguage.ARABIC,
            }

            detected_lang = lang_mapping.get(lang_part, SupportedLanguage.ENGLISH)

            # Map to regions (simplified)
            region_mapping = {
                "US": Region.NORTH_AMERICA,
                "CA": Region.NORTH_AMERICA,
                "GB": Region.EUROPE,
                "DE": Region.EUROPE,
                "FR": Region.EUROPE,
                "IT": Region.EUROPE,
                "ES": Region.EUROPE,
                "JP": Region.ASIA_PACIFIC,
                "KR": Region.ASIA_PACIFIC,
                "CN": Region.ASIA_PACIFIC,
                "AU": Region.ASIA_PACIFIC,
                "BR": Region.LATIN_AMERICA,
                "MX": Region.LATIN_AMERICA,
                "AR": Region.LATIN_AMERICA,
            }

            detected_region = region_mapping.get(region_part, Region.NORTH_AMERICA)

            # Get timezone from environment
            timezone = os.environ.get("TZ", "UTC")

            return LocalizationConfig(
                language=detected_lang,
                region=detected_region,
                timezone=timezone
            )

        except Exception as e:
            self.logger.warning(f"Failed to detect locale from environment: {e}")
            return LocalizationConfig()  # Return default

    def export_messages_template(self, output_path: str, language: SupportedLanguage) -> None:
        """Export message template for translation."""
        template = {}

        for key, translations in self.message_catalog.messages.items():
            # Use English as template base
            english_message = translations.get("en", key)
            template[key] = {
                "original": english_message,
                "translation": translations.get(language.value, ""),
                "context": f"Message key: {key}",
                "needs_translation": language.value not in translations
            }

        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(template, f, indent=2, ensure_ascii=False)

            self.logger.info(f"Exported translation template to {output_path}")

        except Exception as e:
            self.logger.error(f"Failed to export translation template: {e}")


# Global localizer instance
_global_localizer: Optional[QuantumLocalizer] = None


def get_localizer() -> QuantumLocalizer:
    """Get global localizer instance."""
    global _global_localizer
    if _global_localizer is None:
        _global_localizer = QuantumLocalizer()
    return _global_localizer


def set_locale(config: LocalizationConfig) -> None:
    """Set global locale configuration."""
    global _global_localizer
    _global_localizer = QuantumLocalizer(config)


def t(key: str, **kwargs) -> str:
    """Global translate function."""
    return get_localizer().get_message(key, **kwargs)


def detect_and_set_locale() -> LocalizationConfig:
    """Detect and set locale from environment."""
    localizer = get_localizer()
    detected_config = localizer.detect_locale_from_environment()
    set_locale(detected_config)
    return detected_config
