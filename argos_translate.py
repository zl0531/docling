import logging
from typing import Optional

_logger = logging.getLogger(__name__)

class ArgosTranslate:
    """Lightweight offline translation using Argos Translate."""
    
    def __init__(self):
        self.initialized = False
        self.from_code = "en"
        self.to_code = "zh"
        self.translation = None
        
    def _initialize_translator(self):
        """Initialize the Argos Translate translator."""
        if self.initialized:
            return
            
        try:
            import argostranslate.package
            import argostranslate.translate
            
            # Update package index
            _logger.info("Updating Argos Translate package index...")
            argostranslate.package.update_package_index()
            
            # Get available packages
            available_packages = argostranslate.package.get_available_packages()
            
            # Check if the package is already installed
            installed_languages = argostranslate.translate.get_installed_languages()
            from_lang = None
            to_lang = None
            
            # Find source and target languages
            for lang in installed_languages:
                if lang.code == self.from_code:
                    from_lang = lang
                if lang.code == self.to_code:
                    to_lang = lang
                    
            # If we found both languages, check if translation is available
            if from_lang and to_lang:
                translation = from_lang.get_translation(to_lang)
                if translation:
                    self.translation = translation
                    self.initialized = True
                    _logger.info(f"Using existing translation from {self.from_code} to {self.to_code}")
                    return
            
            # If translation not available, look for package to install
            _logger.info("Looking for translation package to install...")
            package_to_install = None
            for package in available_packages:
                if package.from_code == self.from_code and package.to_code == self.to_code:
                    package_to_install = package
                    break
            
            # If not found, try alternative Chinese codes
            if package_to_install is None:
                for package in available_packages:
                    if package.from_code == self.from_code and (
                        package.to_code.startswith("zh") or "chinese" in package.to_code.lower()
                    ):
                        package_to_install = package
                        self.to_code = package.to_code
                        _logger.info(f"Found alternative Chinese package: {self.to_code}")
                        break
            
            if package_to_install is None:
                raise Exception(f"No translation package found for {self.from_code} to {self.to_code}")
            
            # Install the package
            _logger.info(f"Installing translation package: {package_to_install}")
            downloaded_path = package_to_install.download()
            argostranslate.package.install_from_path(downloaded_path)
            
            # Reload installed languages
            installed_languages = argostranslate.translate.get_installed_languages()
            from_lang = None
            to_lang = None
            
            # Find source and target languages again
            for lang in installed_languages:
                if lang.code == self.from_code:
                    from_lang = lang
                if lang.code == self.to_code:
                    to_lang = lang
                    
            if from_lang and to_lang:
                translation = from_lang.get_translation(to_lang)
                if translation:
                    self.translation = translation
                    self.initialized = True
                    _logger.info("Argos Translate initialized successfully")
                    return
            
            raise Exception("Failed to initialize translation after package installation")
            
        except ImportError as e:
            _logger.error(f"Argos Translate not installed: {e}")
            raise Exception("Argos Translate package not installed. Please run 'pip install argostranslate'")
        except Exception as e:
            _logger.error(f"Error initializing Argos Translate: {e}")
            raise
    
    def translate(self, text: str, src: str = "en", dest: str = "zh") -> str:
        """Translate text using Argos Translate.
        
        Args:
            text: Text to translate
            src: Source language (default: en)
            dest: Target language (default: zh for Chinese)
            
        Returns:
            Translated text
        """
        if not text.strip():
            return text
            
        try:
            # Initialize translator if not already done
            if not self.initialized:
                # Update source and target codes
                self.from_code = src
                self.to_code = dest
                self._initialize_translator()
            
            # Translate the text
            translated_text = self.translation.translate(text)
            return translated_text
            
        except Exception as e:
            _logger.error(f"Error translating text with Argos Translate: {e}")
            return text