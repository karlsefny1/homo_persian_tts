"""Run inside python-sandbox: python -m unittest discover -s tests -v."""
import unittest
from unittest.mock import Mock

from homograph_text import strip_diacritics
from persian_chatterbox_tts import (
    PersianChatterboxPipeline, check_diacritic_tokens, split_text,
)


class WorkflowTests(unittest.TestCase):
    def test_classifier_output_is_sent_to_tts(self):
        original = "او اتاق را ترک کرد."
        prepared = "او اتاق را تَرْک کرد."
        decisions = [{"token": "ترک", "changed": True}]
        bert = Mock()
        bert.analyze.return_value = (prepared, decisions)
        pipeline = PersianChatterboxPipeline(diacritizer=bert)
        pipeline.render = Mock(return_value=("waveform", 24000))
        result = pipeline.synthesize(original, "reference.wav", seed=7)
        bert.analyze.assert_called_once_with(original)
        pipeline.render.assert_called_once_with(prepared, "reference.wav", seed=7)
        self.assertEqual(result, ("waveform", 24000, prepared, decisions))

    def test_classifier_cannot_rewrite_words(self):
        bert = Mock()
        bert.analyze.return_value = ("او خانه را ترک کرد.", [])
        pipeline = PersianChatterboxPipeline(diacritizer=bert)
        pipeline.render = Mock()
        with self.assertRaisesRegex(RuntimeError, "other than diacritics"):
            pipeline.synthesize("او اتاق را ترک کرد.", "reference.wav")
        pipeline.render.assert_not_called()

    def test_blank_input_does_not_load_models(self):
        pipeline = PersianChatterboxPipeline()
        with self.assertRaises(ValueError):
            pipeline.prepare(" \n\t")
        self.assertIsNone(pipeline.diacritizer)
        self.assertIsNone(pipeline.tts)

    def test_chunking_preserves_text_and_marks(self):
        text = "  او اتاق را تَرْک کرد.\n\n" + "این یک متن فارسی با نیم‌فاصله است. " * 8
        chunks = split_text(text, 70)
        self.assertEqual("".join(chunks), text)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(c.rstrip()) <= 70 for c in chunks))
        for chunk in chunks:
            self.assertFalse(chunk.lstrip().startswith(("َ", "ْ")))

    def test_huge_word_is_not_truncated(self):
        with self.assertRaisesRegex(ValueError, "word exceeds"):
            split_text("ک" * 300)

    def test_tokenizer_dropping_marks_is_rejected(self):
        tokenizer = Mock()
        tokenizer.encode.return_value = [1, 2]
        with self.assertRaisesRegex(RuntimeError, "discarded"):
            check_diacritic_tokens(tokenizer, "تَرْک")

    def test_unknown_marks_are_rejected_even_if_ids_differ(self):
        tokenizer = Mock()
        tokenizer.encode.side_effect = [[1, 99, 2], [1, 2]]
        tokenizer.tokenizer.decode.return_value = "ت[UNK]رک"
        with self.assertRaisesRegex(RuntimeError, "preserve"):
            check_diacritic_tokens(tokenizer, "تَرْک")

    def test_preserved_marks_accepted_without_language_prefix(self):
        tokenizer = Mock()
        tokenizer.encode.side_effect = [[1, 4, 2, 5, 3], [1, 2, 3]]
        tokenizer.tokenizer.decode.return_value = "ت َ ر ْ ک"
        check_diacritic_tokens(tokenizer, "تَرْک")
        self.assertEqual(tokenizer.encode.call_args_list[0].kwargs, {"language_id": None})


if __name__ == "__main__":
    unittest.main()
