import re
import emoji.core as emoji


class EmojiMap:
    def __init__(self):
        self.emoji_to_text_map = {}
        self.text_to_emoji_map = {}
        self.start_token = 1
        self.emoji_class = emoji

    def add_emoji_to_text_map(self, emoji, text):
        self.emoji_to_text_map[emoji] = text
        self.text_to_emoji_map[text] = emoji

    def process_text(self, text):
        all_emojis_in_text = self.emoji_class.emoji_list(text)
        for emoji in all_emojis_in_text:
            emoji_id = "emoji" + str(self.start_token)
            emoji_id = emoji_id.strip()
            self.add_emoji_to_text_map(emoji["emoji"], emoji_id)
            text = text.replace(emoji["emoji"], " " + emoji_id + " ")
            self.start_token += 1
        return text

    def decode_text(self, text):
        reg_exp = r"emoji\d+"
        emoji_list = re.findall(reg_exp, text)
        for emj in emoji_list:
            text = text.replace(emj, self.text_to_emoji_map[emj])
        return text

    def decode_text_doc(self, text):
        reg_exp = r"emoji\d+"
        emoji_list = re.findall(reg_exp, text)
        for emj in emoji_list:
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


