"""
Claude Prompt Generator - Python Implementation
Converts the HTML-based prompt generator to a Python function.
"""

import json
from typing import Dict, Any, Optional, Union


def generate_claude_prompt(
    words_per_topic_json: Dict[str, Any],
    docs_per_topic_json: Dict[str, Any],
    prompt_style: int,
    language: str = 'tr',
    custom_format: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generate a Claude prompt from topic analysis JSON data.
    
    Args:
        words_per_topic_json: Dictionary containing word scores per topic
        docs_per_topic_json: Dictionary containing document scores per topic  
        prompt_style: Integer 1-5 for prompt style (1=standard, 2=detailed, 3=simple, 4=numbered, 5=custom)
        language: Language code (tr, en, es, fr, de, it, pt, ru, zh, ja, ko, ar)
        custom_format: Custom format string for style 5
        
    Returns:
        Dictionary with generated prompt and metadata
    """
    
    try:
        # Validate inputs
        if not words_per_topic_json or not docs_per_topic_json:
            return {
                "prompt": "",
                "language": language,
                "style": "error",
                "success": False,
                "error": "Both JSON inputs are required"
            }
        
        if prompt_style not in range(1, 6):
            return {
                "prompt": "",
                "language": language,
                "style": "error", 
                "success": False,
                "error": "Prompt style must be between 1-5"
            }
            
        if prompt_style == 5 and not custom_format:
            return {
                "prompt": "",
                "language": language,
                "style": "error",
                "success": False,
                "error": "Custom format is required for style 5"
            }

        # Format templates matching the HTML version
        format_templates = {
            1: "**Konu {SAYI}: {BAŞLIK}**\n- {AÇIKLAMA}",
            2: "## Konu {SAYI}: {BAŞLIK}\n**Ana Tema:** {TEMA}\n**Açıklama:** {AÇIKLAMA}\n**Durum:** {DURUM}",
            3: "• Konu {SAYI}: {BAŞLIK}",
            4: "{SAYI}. **{BAŞLIK}**\n   {AÇIKLAMA}",
            5: custom_format
        }
        
        # Language settings matching the HTML version
        language_settings = {
            'tr': {
                'name': 'Türkçe',
                'instruction': 'Türkçe olsun'
            },
            'en': {
                'name': 'English', 
                'instruction': 'Write in English'
            },
            'es': {
                'name': 'Español',
                'instruction': 'Escribe en español'
            },
            'fr': {
                'name': 'Français',
                'instruction': 'Écris en français'
            },
            'de': {
                'name': 'Deutsch',
                'instruction': 'Schreibe auf Deutsch'
            },
            'it': {
                'name': 'Italiano',
                'instruction': 'Scrivi in italiano'
            },
            'pt': {
                'name': 'Português',
                'instruction': 'Escreva em português'
            },
            'ru': {
                'name': 'Русский',
                'instruction': 'Пиши на русском языке'
            },
            'zh': {
                'name': '中文',
                'instruction': '用中文写'
            },
            'ja': {
                'name': '日本語',
                'instruction': '日本語で書いてください'
            },
            'ko': {
                'name': '한국어',
                'instruction': '한국어로 작성하세요'
            },
            'ar': {
                'name': 'العربية',
                'instruction': 'اكتب باللغة العربية'
            }
        }
        
        # Validate language
        if language not in language_settings:
            return {
                "prompt": "",
                "language": language,
                "style": "error",
                "success": False,
                "error": f"Unsupported language: {language}"
            }
        
        # Get format template and language settings
        format_template = format_templates[prompt_style]
        lang_setting = language_settings[language]
        
        # Style names for return value
        style_names = {
            1: "standard",
            2: "detailed", 
            3: "simple",
            4: "numbered",
            5: "custom"
        }
        
        # Generate the prompt (matching HTML version exactly)
        prompt = f"""Aşağıdaki iki JSON verisini kullanarak konular için başlık üret:
1. DOKÜMANLAR JSON'ı: Her konunun en önemli yorumları/dokümanları
2. KELİMELER JSON'ı: Her konunun en önemli anahtar kelimeleri ve skorları

ÇIKTI FORMATI (tam olarak bu şekilde):
{format_template}

KURALLAR:
- Her konu için tek başlık üret
- Başlıkları hem anahtar kelimeleri hem de dokümanlardaki içeriği dikkate alarak oluştur
- Başlıklar net, kısa ve özetleyici olsun
- Kullanıcı davranışlarını ve ana temaları yansıtsın
- {lang_setting['instruction']}
- Verilen formatı kesinlikle değiştirme
- Konu sayısı her iki JSON'daki konu sayısıyla aynı olsun
- Yüksek skorlu kelimeleri öncelikle dikkate al

DİL: {lang_setting['name']}

DOKÜMANLAR JSON'ı (En önemli yorumlar):
{json.dumps(docs_per_topic_json, ensure_ascii=False, indent=2)}

KELİMELER JSON'ı (Anahtar kelimeler ve skorları):
{json.dumps(words_per_topic_json, ensure_ascii=False, indent=2)}

Lütfen bu konular için başlıkları {lang_setting['name']} dilinde ve belirttiğim formatta üret. Her konunun hem dokümanlarını hem de anahtar kelimelerini analiz ederek anlamlı başlıklar oluştur."""

        return {
            "prompt": prompt,
            "language": lang_setting['name'],
            "style": style_names[prompt_style],
            "success": True,
            "error": None
        }
        
    except Exception as e:
        return {
            "prompt": "",
            "language": language,
            "style": "error",
            "success": False,
            "error": f"Error generating prompt: {str(e)}"
        }


def generate_claude_prompt_single_json(
    json_data: Dict[str, Any],
    prompt_style: int,
    language: str = 'tr', 
    custom_format: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generate a Claude prompt from single JSON data (legacy support).
    
    Args:
        json_data: Dictionary containing topic data
        prompt_style: Integer 1-5 for prompt style
        language: Language code
        custom_format: Custom format string for style 5
        
    Returns:
        Dictionary with generated prompt and metadata
    """
    
    try:
        # Validate inputs
        if not json_data:
            return {
                "prompt": "",
                "language": language,
                "style": "error",
                "success": False,
                "error": "JSON data is required"
            }
        
        if prompt_style not in range(1, 6):
            return {
                "prompt": "",
                "language": language,
                "style": "error",
                "success": False, 
                "error": "Prompt style must be between 1-5"
            }
            
        if prompt_style == 5 and not custom_format:
            return {
                "prompt": "",
                "language": language,
                "style": "error",
                "success": False,
                "error": "Custom format is required for style 5"
            }

        # Format templates
        format_templates = {
            1: "**Konu {SAYI}: {BAŞLIK}**\n- {AÇIKLAMA}",
            2: "## Konu {SAYI}: {BAŞLIK}\n**Ana Tema:** {TEMA}\n**Açıklama:** {AÇIKLAMA}\n**Durum:** {DURUM}",
            3: "• Konu {SAYI}: {BAŞLIK}",
            4: "{SAYI}. **{BAŞLIK}**\n   {AÇIKLAMA}",
            5: custom_format
        }
        
        # Language settings
        language_settings = {
            'tr': {'name': 'Türkçe', 'instruction': 'Türkçe olsun'},
            'en': {'name': 'English', 'instruction': 'Write in English'},
            'es': {'name': 'Español', 'instruction': 'Escribe en español'},
            'fr': {'name': 'Français', 'instruction': 'Écris en français'},
            'de': {'name': 'Deutsch', 'instruction': 'Schreibe auf Deutsch'},
            'it': {'name': 'Italiano', 'instruction': 'Scrivi in italiano'},
            'pt': {'name': 'Português', 'instruction': 'Escreva em português'},
            'ru': {'name': 'Русский', 'instruction': 'Пиши на русском языке'},
            'zh': {'name': '中文', 'instruction': '用中文写'},
            'ja': {'name': '日本語', 'instruction': '日本語で書いてください'},
            'ko': {'name': '한국어', 'instruction': '한국어로 작성하세요'},
            'ar': {'name': 'العربية', 'instruction': 'اكتب باللغة العربية'}
        }
        
        # Validate language
        if language not in language_settings:
            return {
                "prompt": "",
                "language": language,
                "style": "error",
                "success": False,
                "error": f"Unsupported language: {language}"
            }
        
        format_template = format_templates[prompt_style]
        lang_setting = language_settings[language]
        
        style_names = {1: "standard", 2: "detailed", 3: "simple", 4: "numbered", 5: "custom"}
        
        # Generate single JSON prompt
        prompt = f"""Aşağıdaki JSON verilerindeki konular için başlık üret. 

ÇIKTI FORMATI (tam olarak bu şekilde):
{format_template}

KURALLAR:
- Her konu için tek başlık üret
- Başlıklar net, kısa ve özetleyici olsun  
- Kullanıcı davranışlarını ve ana temaları yansıtsın
- {lang_setting['instruction']}
- Verilen formatı kesinlikle değiştirme
- Başlık sayısı JSON'daki konu sayısıyla aynı olsun

DİL: {lang_setting['name']}

JSON VERİSİ:
{json.dumps(json_data, ensure_ascii=False, indent=2)}

Lütfen bu konular için başlıkları {lang_setting['name']} dilinde ve belirttiğim formatta üret."""

        return {
            "prompt": prompt,
            "language": lang_setting['name'],
            "style": style_names[prompt_style],
            "success": True,
            "error": None
        }
        
    except Exception as e:
        return {
            "prompt": "",
            "language": language,
            "style": "error",
            "success": False,
            "error": f"Error generating prompt: {str(e)}"
        }


# Example usage and testing
if __name__ == "__main__":
    # Example data for testing
    sample_words_json = {
        "Konu 00": {
            "uygulama": 5.123,
            "kullanıcı": 4.567,
            "memnun": 3.456
        },
        "Konu 01": {
            "sorun": 4.789,
            "hata": 3.234,
            "düzeltme": 2.876
        }
    }
    
    sample_docs_json = {
        "Konu 00": {
            "doc1": "Bu uygulama gerçekten harika, çok memnunum:0.95",
            "doc2": "Kullanıcı deneyimi mükemmel:0.87"
        },
        "Konu 01": {
            "doc1": "Uygulama sürekli hata veriyor:0.92", 
            "doc2": "Sorunları bir an önce düzeltin:0.89"
        }
    }
    
    # Test the function
    result = generate_claude_prompt(
        words_per_topic_json=sample_words_json,
        docs_per_topic_json=sample_docs_json,
        prompt_style=1,
        language='tr'
    )
    
    print("Test Result:")
    print(f"Success: {result['success']}")
    print(f"Language: {result['language']}")
    print(f"Style: {result['style']}")
    print("\nGenerated Prompt:")
    print(result['prompt'])