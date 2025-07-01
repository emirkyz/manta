import re

try:
    import emoji
    EMOJI_AVAILABLE = True
except ImportError:
    EMOJI_AVAILABLE = False
except Exception as e:
    # Handle Django configuration error or other import issues
    print(f"Warning: Emoji library not properly configured: {e}")
    EMOJI_AVAILABLE = False


class EmojiMap:
    def __init__(self):
        self.emoji_to_text_map = {}
        self.text_to_emoji_map = {}
        self.start_token = 1
        self.emoji_available = EMOJI_AVAILABLE
        
        if EMOJI_AVAILABLE:
            try:
                # Test if emoji library is working properly
                emoji.emoji_count("test")
                self.emoji_class = emoji
            except Exception as e:
                print(f"Warning: Emoji library not working properly: {e}")
                self.emoji_available = False
                self.emoji_class = None
        else:
            self.emoji_class = None
    
    def add_emoji_to_text_map(self, emoji_char, text):
        self.emoji_to_text_map[emoji_char] = text
        self.text_to_emoji_map[text] = emoji_char
        
    def process_text(self, text):
        if not self.emoji_available or self.emoji_class is None:
            # If emoji library is not available, use a simple regex-based approach
            return self._fallback_process_text(text)
            
        try:
            all_emojis_in_text = self.emoji_class.emoji_list(text)
            for emoji_data in all_emojis_in_text:
                emoji_id = "emoji" + str(self.start_token)
                emoji_id = emoji_id.strip()
                self.add_emoji_to_text_map(emoji_data["emoji"], emoji_id)
                text = text.replace(emoji_data["emoji"], " " + emoji_id + " ")
                self.start_token += 1
            return text
        except Exception as e:
            print(f"Warning: Error processing emojis with emoji library: {e}")
            return self._fallback_process_text(text)

    def _fallback_process_text(self, text):
        """Fallback method to handle emojis when emoji library is not available"""
        # Simple regex to find emoji-like characters (Unicode ranges for emojis)
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F1E0-\U0001F1FF"  # flags (iOS)
            "\U00002702-\U000027B0"
            "\U000024C2-\U0001F251"
            "]+", flags=re.UNICODE
        )
        
        def replace_emoji(match):
            emoji_char = match.group()
            emoji_id = "emoji" + str(self.start_token)
            emoji_id = emoji_id.strip()
            self.add_emoji_to_text_map(emoji_char, emoji_id)
            self.start_token += 1
            return " " + emoji_id + " "
        
        return emoji_pattern.sub(replace_emoji, text)

    def decode_text(self, text):
        reg_exp = r"emoji\d+"
        emoji_list = re.findall(reg_exp, text)
        for emj in emoji_list:
            if emj in self.text_to_emoji_map:
                text = text.replace(emj, self.text_to_emoji_map[emj])
        return text

    def decode_text_doc(self, text):
        reg_exp = r"emoji\d+"
        emoji_list = re.findall(reg_exp, text)
        for emj in emoji_list:
            if emj in self.text_to_emoji_map:
                text = text.replace(emj, self.text_to_emoji_map[emj])
        return text
    
    def check_if_text_contains_tokenized_emoji(self, text):
        reg_exp = r"emoji\d+"
        emoji_list = re.findall(reg_exp, text)
        if len(emoji_list) > 0:
           return True
        return False

    def check_if_text_contains_tokenized_emoji_doc(self, text):
        reg_exp = r'\bemoji\d+\b'
        emoji_list = re.findall(reg_exp, text)
        if len(emoji_list) > 0:
           return True
        return False


